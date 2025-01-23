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
from typing import Optional, Union, List, Set, TypeVar, Any, Dict, KeysView, ItemsView
from modelflow.utils.helper import singleton
import threading
from hydra.utils import instantiate
from functools import wraps
import inspect
from modelflow import CFG

log = logging.getLogger(__name__)

Sto = TypeVar("Sto", bound="Store")
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
          r.union(_find_missing_variables(s))
    else:
        r.union(_find_missing_variables(s))
    return r

def expand_variables(template_str, **variables):
    template = StringTemplate(template_str)
    return template.safe_substitute(**variables)

def expand_parameters(datastore:Sto):
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

@dataclass
class Task(abc.ABC):
    outputs: Optional[List['Output']]
    task_ouput: Any = None # holds arbitrary data from task, can be ephemeral
    def __post_init__(self):
        self.started_at = None
        self.completed_at = None
        self.success = None
        
    def run_outputs(self):
        for output in self.outputs:
            output()
        pass
    
    def complete(self, success):
        self.completed_at = datetime.datetime.now()
        self.success = success
    
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass
    
    def __call__(self,  *args, **kwargs):
        return self.run(args, kwargs)

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
        pass
    
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
        return self.update(*args, **kwargs)

class Store(abc.ABC):
    """
    Base class for a variable Store. Holds inputs and outputs of commands and pipeline steps.
    Observers can be registered to listen to key updates. Not a dataclass so we can put multiple decorators around subclasses.
    """
    def __init__(self):
        # Holds any data, accessable by str keys, should always stay this way 
        self._data: Dict[str, Any] = {}
        self._observer: Dict[str, Any] = {}
        
    def update(self, key, value):
        self._data[key] = value
        self._notify(key, value)

    def keys(self) -> KeysView[Any]:
        return self._data.keys()
    
    def items(self) -> ItemsView[Any, Any]:
        return self._data.items()
    
    def __contains__(self, value) -> bool:
        return self.__dict__.keys().__contains__(value)
    
    def get(self, key, default) -> Any:
        return self._data.get(key, default=default)
    
    def _notify(self, key, value):
        if key in self._observers:
            for observer in self._observers[key]:
                observer.update(key, value)
    
    def register_observer(self, key, observer):
        if key not in self._observers:
            self._observers[key] = []
        self._observers[key].append(observer)

    def unregister_observer(self, key, observer):
        if key in self._observers:
            self._observers[key].remove(observer)
            if not self._observers[key]:
                del self._observers[key]

@dataclass
class MetaTaskList(Task):
    common_cmd_args: str = None
    tasks: List[Union[Tas, Com]] = field(default_factory=lambda: [])
    
@dataclass
class Sequence(MetaTaskList):
    
    def run(self, *args, **kwargs):
        for cmd in self.tasks:
            cmd.run(*args, **kwargs)

@dataclass
class Parrallel(MetaTaskList):

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
        to = os.path.expandvars(path=to)
        to = os.path.expanduser(path=to)
        os.makedirs(to, exist_ok=True)
        self.to = to
    
    def run(self, *args, **kwargs):
        value = None # comes from source object, Goal should get called with a source object or Command as ref 
        if "value" in kwargs.keys():
            # dangerous but i think we will only ever have one output from a source called value
            # we could also do this by kwargs then...
            value = kwargs["value"] 
        if self.store is None:
            self.store = OutputManager() # singleton is also global, so we can just use this here (dirty but works i guess)
        self.store.update_value(self.key, value)

@singleton()
class OutputManager(Store):
    """
    Holds Information about Command Outputs and notifies outputs to subscribed keys in dict when a new value gets written to the output.
    """
    def __init__(self):
        super().__init__()
        log.info("creating output store")
        pass