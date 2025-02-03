import abc
import os
from os import PathLike
import time
import datetime
import subprocess
from string import Template
import logging
import re
from dataclasses import dataclass, field
from typing import Optional, Union, List, Set, TypeVar, Any, Dict, KeysView, ItemsView, ClassVar
from modelflow.utils.helper import singleton
import threading
from hydra.utils import instantiate
from functools import wraps
import inspect
from modelflow import CFG
from modelflow.store.store import OutputManager

log = logging.getLogger(__name__)

Tas = TypeVar('Tas', bound='Task') # has to be callable 
Com = TypeVar('Com', bound='Command') # has to be callable 


class CallerClassNotFoundException(Exception):
    """Exception for when the caller class gets not found."""
    def __init__(self, message):
        super().__init__(message)

    
class OutputTimeoutException(Exception):
    """Exception raised when an Output's timeout is reached without an update."""
    def __init__(self, message):
        super().__init__(message)
        

class RegexStringNotFoundException(Exception):
    def __init__(self, message):
        super().__init__(message)

def find_caller_class(*target_class, raise_if_notfound=True):
    """
    Searches the call stack for an instance of the specified class.

    Parameters:
    - target_class: The class to search for in the call stack.

    Returns:
    A message indicating whether the class was found and in which function.
    """
    for frame_info in inspect.stack():
        # Check if 'self' exists in the current frame's local variables
        if 'self' in frame_info.frame.f_locals:
            # Get the instance (self) from the frame
            instance = frame_info.frame.f_locals['self']
            # Check if instance is of target_class
            if isinstance(instance, target_class):
                log.debug(f"Found Target Caller in frame {instance.__name__}")
                return instance
    if raise_if_notfound:
        log.warning(f"{target_class} not found in any frame.")
    else:
        raise CallerClassNotFoundException(f"Class {target_class} not found in call stack")

class StringTemplate(Template):
    delimiter = '{{'
    pattern = r'''
    \{\{(?:
        (?P<escaped>\{\{)|
        (?P<named>[^\{\}\s]+)\}\}|
        (?P<braced>[^\{\}\s]+)\}\}|
        (?P<invalid>)
    )
    '''
    
def find_missing_variables(template_str: Union[List[str], str]) -> Set[str]:
    def _find_missing_variables(template_s:str):
        template = StringTemplate(template_s)
        expected_vars = set(re.findall(template.pattern, template_s))
        # Flatten the set of tuples and filter out non-variable patterns (like escaped '{{')
        expected_vars = {match[1] or match[2] for match in expected_vars if match[1] or match[2]}
        return expected_vars
    r = set()
    if isinstance(template_str, list):
       for s in template_str:
          r = r.union(_find_missing_variables(s))
    else:
        r = r.union(_find_missing_variables(s))
    return r

def expand_variables(template_str, **variables):
    template = StringTemplate(template_str)
    return template.safe_substitute(**variables)

def expand_parameters(filter=[]):
    datastore = OutputManager()
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # Retrieve the method's signature
            sig = inspect.signature(method)
            bound_args = sig.bind(self, *args, **kwargs)
            bound_args.apply_defaults()
            
            new_args = list(args)
            # Update args and kwargs with expanded variables from the datastore
            for index, (name, value) in enumerate(bound_args.arguments.items()):
                if len(filter) > 0 and name not in filter:
                    continue
                
                if name == 'self':
                    continue
                
                if isinstance(value, str):  # Only expand if value is a string
                    
                    if name in datastore:
                        # If the parameter is positional
                        if index-1 < len(args):
                            new_args[index-1] = expand_variables(value, **datastore)
                        # If the parameter is keyword
                        else:
                            kwargs[name] = expand_variables(value, **datastore)
                    else:
                        # Expand with available datastore variables even if not explicitly in datastore
                        if index-1 < len(args):
                            new_args[index-1] = expand_variables(value, **datastore)
                        else:
                            kwargs[name] = expand_variables(value, **datastore)

            return method(self, *new_args, **kwargs)
        return wrapper
    return decorator

def expand_class_attributes(filter=[]):
    datastore = OutputManager()
    def decorator(method):
        @wraps(method)
        def wrapper(self, *args, **kwargs):
            # First, expand the class attributes using the given datastore
            for attr_name, attr_value in vars(self).items():
                if len(filter) > 0 and attr_name not in filter:
                    continue
                
                if isinstance(attr_value, str):
                    setattr(self, attr_name, expand_variables(attr_value, **datastore))
            
            # Now call the original method
            return method(self, *args, **kwargs)
        return wrapper
    return decorator

@dataclass
class Task(abc.ABC):
    outputs: Optional[List['Output']] = None
    name: Union[str, None] = None
    _task_Counter: ClassVar[list[int]] = 0
    # task_ouput: Any = None # holds arbitrary data from task, can be ephemeral
    def __post_init__(self):
        self.started_at = None
        self.completed_at = None
        self.success = None
        if self.name is None:
            self.name = f"{self.__class__}{self.__class__._task_Counter}"
            self.__class__._task_Counter = self.__class__._task_Counter + 1
        
    def run_outputs(self):
        for output in self.outputs:
            output()
        pass
    
    def complete(self, success):
        self.completed_at = datetime.datetime.now()
        self.success = success
    
    def is_completed(self):
        return self.success is not None
    
    def get_success(self):
        return self.success
    
    def run(self, scheduler, *args, **kwargs):
        return scheduler.run(*args, **kwargs) 
    
    def __call__(self,  *args, **kwargs):
        return self.run(args, kwargs)

    def __repr__(self):
        return f"{self.__class__}:{self.name}"

    def __str__(self):
        return f"{self.__class__}:{self.name}"
    
@dataclass
class Command(Task):
    cmd_args: str = None
    
@dataclass
class SystemCommand(Command):
    cmd_bin: str = None
    
    def subprocess_run(self, shell=bool):
        result = None
        try:
            result: subprocess.CompletedProcess = subprocess.run(self.cmd_bin + " " + self.cmd_args, shell=shell, check=True)    
        except subprocess.CalledProcessError as e:
            log.exception(f"subprocess ended with error {e}")
        return result

    def run(self):
        return self.subprocess_run(True)
    
class VariableMeta(abc.ABC):
    @abc.abstractmethod
    def get_depended_vars(self)-> Set[str]:
        """returns variables that this Class depends on. Always return a Set for API confirmaty"""
        raise NotImplementedError

@dataclass
class Goal(VariableMeta):
    def get_depended_vars(self):
        raise NotImplementedError
    
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass   
    def __call__(self,  *args, **kwargs):
        return self.run(args, kwargs)

@dataclass
class Source(VariableMeta):
    """Source for whatever the saver tries to save inot a store or to a file"""
    # value: str = None # could be a variable or file path, meaning depends on impl-
    def __post_init__(self):
        self.value = None
    
    def get_depended_vars(self):
        """returns variables that this Class depends on"""
        # return {v for set_of_v in self.values for v in find_missing_variables(set_of_v)}
        return find_missing_variables(self.value)
    
    def get_value(self):
        """return precess output based on value and the type of comannd"""
        return self.value
       
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass

    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

Sou = TypeVar("Sou", bound=Source)
Goa = TypeVar("Goa", bound=Goal)
Com = TypeVar("Com", bound=Command)
#Out = TypeVar("Out", bound=Output)
@dataclass
class Output():
    """Outputs handle the command outputs and potentially saves some data to files or inside a variable"""
    source: Any = None
    goals: Any = None
    timeout: int = 60  # Timeout in seconds after command execution has ended and there is no update
    _timer_thread = None
    completed = False
    
    def reset_completion(self):
        self.completed = False
        
    def update(self, *args, **kwargs) -> Any:
        """logic to call an update, either after some dependen variable was updated in the VariableStore or after a Trigger event"""
        if not self.completed:
            self.completed = True
            return self.updateOutput(*args, *kwargs)
        else:
            return None

    def updateOutput(self, key) -> Any:
        """
        Called when the output key is updated. This should finish the output and write whatever output to the OutputManager.
        Override this method in subclasses if necessary, otherwise standard logic just calls 
        """
        log.info(f"Output {self} updated with key {key}")
    
    def start_timeout(self):
        """
        Start the timeout after the command is done.
        """
        # After the command finishes, reset the timer
        if self._timer_thread is not None:
            self._timer_thread.cancel()
        self._timer_thread = threading.Timer(self.timeout, self.on_timeout)
        self._timer_thread.start()
    
    def on_timeout(self):
        """ So far: If a Output does not find an output the pipeline breaks """
        raise OutputTimeoutException(f"Timeout reached without update for Observer {self} on command {self.command}")
    
    def __call__(self, *args, **kwargs):
        log.info(f"Running output {self}")
        return self.update(*args, **kwargs)

@dataclass
class TaskInterator(Task):
    """
    Task abstraction to hold a sequence of Tasks implemented as an iterator.
     
    """
    common_cmd_args: str = ""
    tasks: List[Task] = field(default_factory=lambda: [])
    
    def is_completed(self):
        return len(self.tasks) <= 0 or any([not t.is_completed() for t in self.tasks])
    
    def __iter__(self): # return iterable, so self in this case as this class implements __next__
        return self

    def __next__(self) -> Task: # Python 2: def next(self)
        # first Task in list is always the one to run by default, you may want to change this in a subclass
        self.__before_task__()
        print("=============")
        print(self.tasks)
        print(type(self.tasks[0]))
        print(self.tasks[0])
        item = next(t for t in self.tasks if not t.is_completed())
        self.__after_task__()
        return item
    
    def __after_task__(self):
        """hook that triggers after __next__ call internally (inside __next__)"""
        pass
    
    def __before_task__(self):
        """hook that triggers before __next__ call internally (inside __next__)"""
        pass
    
    def run(self, scheduler, *args, **kwargs):
        """not recommended calling chain, use run of scheudler please"""
        log.warning("not recommended calling chain, use run of scheudler please")
        return scheduler.run(*args, **kwargs) 

    def __str__(self):
        return f"{self.__class__}:{self.name}-{self.tasks}"
    
@dataclass
class Sequence(TaskInterator):
    
    def run(self, *args, **kwargs):
        for cmd in self.tasks:
            cmd.run(*args, **kwargs)

@dataclass
class Parrallel(TaskInterator):

    def run(self, *args, **kwargs):
        raise NotImplementedError
    def __call__(self,  *args, **kwargs):
        return self.run(args, kwargs)
############## Impl ###################

@dataclass
class StdoutSource(Source):
    regex: str # regex to extract from string
    def __post_init__(self):
        super().__post_init__()
        self.pattern = re.compile(self.regex)
        
    def get_task_output(self, *classes):
        relevant_caller = find_caller_class(*classes)
        return relevant_caller.task_output
    
    def run(self):
        # expand_variables(self.path, ) # variable interpolation might break regex?
        task_ouput = self.get_task_output(Command, Task)
        match = self.pattern.match(task_ouput)
        extracted_string = None
        if match:
            extracted_string = match.group(1)
            log.info(f"{extracted_string} extracted from task output")
        else:
            raise RegexStringNotFoundException(f"Could not extract {self.regex} from task output")
            
        self.value = extracted_string
        #maybe we could do var expansion here? prob not necessary, if so this has to be a parameter
        return self.value
        
@dataclass
class FilePathSource(Source):
    path: str
    def get_depended_vars(self)-> Set[str]:
        raise NotImplementedError
    
    def run(self):
        self.value = expand_variables(self.path, ) # variable interpolation!

@dataclass
class StoreVariableGoal(Goal):
    key: str = None
    value: Any = None
    store: Any = None
    # def __post__init(self):
    #     if self.store is None:
    #         self.store = CFG()
    
    def run(self, *args, **kwargs):
        value = self.value
        if value is None:
            # dangerous but i think we will only ever have one output from a source called value
            # we could also do this by kwargs then...
            value = kwargs["value"] 
        if self.store is None:
            self.store = OutputManager() # singleton is also global, so we can just use this here (dirty but works i guess)
        self.store.update_value(self.key, value)
    
@dataclass
class FileCopyGoal(Goal):
    to: Any = None
    def __post_init__(self):
        to = self.to
        to = os.path.expandvars(path=self.to)
        to = os.path.expanduser(path=to)
        self.to = to
                 
    @expand_class_attributes(filter=["to"])
    def run(self, *args, **kwargs):
        value = None # comes from source object, Goal should get called with a source object or Command as ref 
        log.info(f"Copying file {value} to {self.to}")
        #datastore = OutputManager()
        #to = expand_variables(self.to, **datastore)
        os.makedirs(self.to, exist_ok=True)
        if "value" in kwargs.keys():
            # dangerous but i think we will only ever have one output from a source called value
            # we could also do this by kwargs then...
            value = kwargs["value"] 
        if self.store is None:
            self.store = OutputManager() # singleton is also global, so we can just use this here (dirty but works i guess)
        self.store.update_value(self.key, value)
