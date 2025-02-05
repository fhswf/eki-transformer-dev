import abc
from typing import Callable, List, Union, Any, Type
import time
import logging
import datetime
from dataclasses import dataclass
import pickle
from modelflow.command.common import Task, TaskInterator

log = logging.getLogger(__name__)

class JobExecutionException(Exception):
    """Exception raised when an Job failed"""
    def __init__(self, message):
        super().__init__(message)


class JobExecutionTimeoutException(Exception):
    """Exception raised when an Job timeout (forcefully)"""
    def __init__(self, message):
        super().__init__(message)
        
class JobConversionException(Exception):
    """Exception raised when an Job failed"""
    def __init__(self, message):
        super().__init__(message)

class Serializable(abc.ABC):
    @abc.abstractmethod
    def get_save_attributes(self):
        """
        Override in subclass to return a list of attribute names to be saved.
        """
        raise NotImplementedError("Must implement get_save_attributes")

    def save_state_to_file(self, filename):
        state = {attr: getattr(self, attr) for attr in self.get_save_attributes()}
        with open(filename, 'wb') as f:
            pickle.dump(state, f)
            
    def load_state_from_file(self, filename):
        with open(filename, 'rb') as f:
            state = pickle.load(f)
        for attr, value in state.items():
            setattr(self, attr, value)
        # Optionally, call a method to handle any post-load initialization
        if hasattr(self, '__after_reload__'):
            self.__post_load__()

    def __after_reload__(self):
        """consider implementing this in subclasses to do post load init type stuff"""
        pass
    
    

@dataclass
class Job(abc.ABC):
    task: Task = None
    breaking_timeout = None # maximum timeout for a job in seconds
    started_at = None
    finished_at = None
    log_interval:int = 60 # log every 60 seconds
    polling_secs:int = 10 # check for job completion every 10 seconds
    def __post_init__(self):
        if self.convert_task_to_job(self.task):
            log.info("job conversion completed")
        else: 
            raise JobConversionException()
        pass
    
    @abc.abstractmethod
    def start(self, *args, **kwargs):
        raise NotImplementedError
    
    def run(self, *args, **kwargs):
        ret = self.start(*args, **kwargs)
        return ret 
            
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)

    @abc.abstractmethod
    def cancel(self, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod 
    def is_completed(self, *args, **kwargs):
        raise NotImplementedError
    
    @abc.abstractmethod 
    def convert_task_to_job(self) -> bool:
        """supposed to fill class attributes with information to acutally run the task"""
        raise NotImplementedError
    
    def wait_for_completion(self, timeout:int=None):
        """blocking loop that waits for is_completed to return True"""
        if timeout is None:
            if self.breaking_timeout is not None:
                timeout = self.breaking_timeout
            else:
                log.warning(f"waiting for job completion for job {self} without timeout")
        while not self.is_completed():
            time.sleep(self.polling_secs)
            if hasattr(self, 'log_interval') and self.log_interval is not None:
                elapsed_time = (datetime.datetime.now() - self.started_at).total_seconds()
                if elapsed_time % self.log_interval < self.polling_secs:
                    log.info(f"Waiting for job {self} to complete. Elapsed time: {elapsed_time} seconds")
            if timeout is not None:
                if  (datetime.now() - self.started_at).total_seconds() > timeout:
                    self.cancel()
                    raise JobExecutionTimeoutException(f"job {self} canceled due to timeout")
        return True
        
    def __repr__(self):
        return f"{self.__class__.__name__} with Task: {self.task}"
    
    def __str__(self):
        return f"{self.__class__.__name__} with Task: {self.task}"
    
@dataclass
class Scheduler(Serializable):
    """Abstract Base class for a shuduler that launches commands. Enables save state loading."""
    policy: Any  = None
    jobClazz: Type[Job] = None

    def __post_init__(self):
        pass
    
    def run(self, maybeTaskInterator: Union[TaskInterator,Task] , *args, **kwargs):
        """launches sheduler"""
        def _run(task):
            job: Job = self.jobClazz(task)
            log.info(f"Scheduler {self.__class__.__name__} Running: {job}")
            job() # runs job.start()
            job.started_at = datetime.datetime.now()
            job.wait_for_completion()
            job.finished_at = datetime.datetime.now()
            log.info(f"Scheduler {self.__class__.__name__} Completed: {job}")
        log.info(f"Scheduler {self.__class__.__name__} started MetaTask: {maybeTaskInterator}")
        if self.policy is not None:
            self.policy.run(maybeTaskInterator, self.jobClazz)
        else:
            if isinstance(maybeTaskInterator, TaskInterator):
                for task in maybeTaskInterator:
                    _run(task)
            elif isinstance(maybeTaskInterator, Task):
                _run(maybeTaskInterator)
            else: 
                raise NotImplementedError(f"{self} TaskInterator or Task expected")
            
    def get_save_attributes(self):
        return ["policy", "jobClazz"]
       
    def __call__(self, taskInterator: TaskInterator, *args, **kwargs):
        return self.run(taskInterator, *args, **kwargs)
    
    def __repr__(self):
        return f"{self.__class__.__name__} with jobClazz: {self.jobClazz}"
    
"""
class Policy(abc.ABC):
    # Runs commands based on the policy. Launch and wait commands can not have arguments atm.
    @abc.abstractmethod
    def execute(self, launch_commands: List[Job], completion_condition: Callable):
        pass
    
class SequentialPolicy(Policy):
    def __init__(self, timeout=None):
        super().__init__()
        self.timeout = timeout # unsed for now
        
    def execute(self, jobs: List[Job], *args, **kwargs):
        for job in jobs:
            job.start(*args, **kwargs)
            log.info(f"Waiting for job {job} to complete...")
            while not job.check():
                time.sleep(10)
                #if self.timeout is not None:
                #    if self.timeout done:
                #        job.cancel()
            log.info(f"Job {log} completed")


class ParallelPolicy(Policy):
    def execute(self, commands, submit_command, is_job_complete):
        job_ids = [submit_command(cmd) for cmd in commands]
        while job_ids:
            for job_id in job_ids[:]:  # Copy the list to avoid modification during iteration
                if is_job_complete(job_id):
                    print(f"Job {job_id} completed")
                    job_ids.remove(job_id)
            time.sleep(10)  # Check every 10 seconds
"""
