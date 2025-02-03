from modelflow.scheduler.common import Scheduler, Job, JobExecutionException
from modelflow.command.common import Task
import re
import time
import logging
import subprocess 
from dataclasses import dataclass
from typing import Type
log = logging.getLogger(__name__)


    
@dataclass
class UnixJob(Job):
    
    def __post_init__(self):
        super().__post_init__()
        self.process = None
        self.command = None 
        
    def start(self):
        log.info(f"Running on local Unix: {self.command}")
        # Start the process without waiting for it to complete
        self.process = subprocess.Popen(self.command, shell=True, capture_output=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)# TODO Read docs
        # completed_at
        
    def cancel(self):
        return self.process.terminate()
        
    def kill(self):
        return self.process.kill()

    def is_completed(self):
        # Use non-blocking I/O on the process to check if it has completed
        if self.process.poll() is not None:  # Process has finished
            self.task.completed_at = time.time()
            self.task.success = (self.process.returncode == 0)
            if self.task.success:
                log.info(f"Unix job {self.command} completed successfully.")
            else:
                log.info(f"Unix job {self.command} failed with status code {self.process.returncode}")
                raise JobExecutionException(f"Unix job {self.command} failed with status code: {self.process.returncode}")
            return True
        return False
    
    def convert_task_to_job(self, task:Task):
        raise NotImplementedError

    def get_process(self):
        return self.process
    
    def get_stdout(self):
        return self.process.stdout
    
    def get_task_output(self):
        return self.get_stdout()

@dataclass

class UnixScheduler(Scheduler):
    """convenience class for local unix execution"""
    
    def __post_init__(self):
        print(self.get_save_attributes())
        pass
    
    def get_save_attributes(self):
        return super().get_save_attributes()