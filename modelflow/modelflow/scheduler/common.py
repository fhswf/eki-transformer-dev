import abc
from typing import Callable, List
import time
import logging
from dataclasses import dataclass
log = logging.getLogger(__name__)

class Scheduler(abc.ABC):
    """Abstract Base class for a shuduler that launches commands. Enables save state loading."""
    def __init__(self):
        raise NotImplementedError
        pass

    @abc.abstractmethod
    def save_sate(self):
        """save sheduler state to drive"""
        raise NotImplementedError
        pass
    
    @abc.abstractmethod
    def load_sate(self):
        """load sheduler from to drive, requires ID or folder?"""
        raise NotImplementedError
        pass
    
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        """launches sheduler"""
        raise NotImplementedError
        pass
    
@dataclass
class Job():
    start: Callable
    cancel: Callable
    check: Callable

    def start(self, *args, **kwargs):
        self.status = self.start(*args, **kwargs)
        pass

    def cancel(self):
        self.cancel(self.status)
        pass
    
    def check(self):
        self.check(self.status)
        pass
    
    @property
    def status(self):
        return self.status

    @status.setter
    def status(self, value):
        self.status = value


class Policy(abc.ABC):
    """Runs commands based on the policy. Launch and wait commands can not have arguments atm."""
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

#class ParallelPolicy(Policy):
#    def execute(self, commands, submit_command, is_job_complete):
#        job_ids = [submit_command(cmd) for cmd in commands]
#        while job_ids:
#            for job_id in job_ids[:]:  # Copy the list to avoid modification during iteration
#                if is_job_complete(job_id):
#                    print(f"Job {job_id} completed")
#                    job_ids.remove(job_id)
#            time.sleep(10)  # Check every 10 seconds