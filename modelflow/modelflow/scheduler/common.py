import abc
from typing import Callable, List, Union
import time
import logging
from dataclasses import dataclass
import pickle
from modelflow.command.common import Task

log = logging.getLogger(__name__)


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
    
    
class Scheduler(Serializable):
    """Abstract Base class for a shuduler that launches commands. Enables save state loading."""
    
    @abc.abstractmethod
    def run(self, run_config:Task, *args, **kwargs):
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



    
    
"""
class ParallelPolicy(Policy):
    def execute(self, commands, submit_command, is_job_complete):
        job_ids = [submit_command(cmd) for cmd in commands]
        while job_ids:
            for job_id in job_ids[:]:  # Copy the list to avoid modification during iteration
                if is_job_complete(job_id):
                    print(f"Job {job_id} completed")
                    job_ids.remove(job_id)
            time.sleep(10)  # Check every 10 seconds

# Scheduler for UNIX processes
class UnixScheduler(Scheduler):
    def run_all(self):
        for task in self.tasks:
            print(f"Running: {task.command}")
            task.started_at = time.time()
            result = subprocess.run(task.command, shell=True, capture_output=True, text=True)  # Note: security implications
            task.completed_at = time.time()
            task.success = (result.returncode == 0)
            if task.success:
                print(f"Output: {result.stdout}")
            else:
                print(f"Error: {result.stderr}")

# Scheduler for Slurm tasks
class SlurmScheduler(Scheduler):
    def run_all(self):
        for task in self.tasks:
            # Command modification for Slurm can be handled here, or within the Task itself
            print(f"Submitting to Slurm: {task.command}")
            task.started_at = time.time()
            # In a real scenario, replace the echo command with sbatch submission
            result = subprocess.run(f"echo 'sbatch {task.command}'", shell=True, capture_output=True, text=True)  # Placeholder for actual sbatch command
            task.completed_at = time.time()
            task.success = (result.returncode == 0)
            if task.success:
                print(f"Slurm job submitted, fake output: {result.stdout}")
            else:
                print(f"Slurm submission error: {result.stderr}")
                
"""
