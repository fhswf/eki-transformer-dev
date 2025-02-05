from modelflow.scheduler.common import Scheduler, Job, JobExecutionException, JobConversionException
from modelflow.command.common import Task, SystemCommand, Command
import re
import time
import logging
import subprocess 
from dataclasses import dataclass
from typing import Type, Any
import shutil
import os
import shlex
log = logging.getLogger(__name__)


    
@dataclass
class UnixJob(Job):
    process: Any = None
    command: Any = None 
    def __post_init__(self):
        super().__post_init__()
        
        
    def start(self):
        log.info(f"Running on local Unix: {self.command}")
        # Start the process without waiting for it to complete
        # Check if the first item in self.command points to an executable
        command_parts = self.command.split()
        executable_path = shutil.which(command_parts[0])

        if executable_path:
            # Replace the first item with the absolute path
            command_parts[0] = executable_path
            self.command = ' '.join(command_parts)
        else:
            log.warning(f"Executable {command_parts[0]} not found in PATH. Attempting to run as is.")
            
        if os.name == 'posix':
            shell = False
            self.command = shlex.split(self.command)
        else:
            shell = True
            # use shell=True on Windows
            # if so args to popen must be a string not array
            
        self.process = subprocess.Popen(self.command, shell=shell, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
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
    
    def convert_task_to_job(self, task: SystemCommand):
        """
        convert SystemCommand to UnixJob. Must return True if successful.
        copies relevant attributes from task into self. 
        """
        if not isinstance(task, SystemCommand):
            raise JobConversionException("task must be an instance of SystemCommand")
        self.task = task
        self.command = task.cmd
        return True

    def get_process(self):
        return self.process
    
    def get_stdout(self):
        return self.process.stdout
    
    def get_task_output(self):
        return self.get_stdout()

    def __repr__(self):
        return f"UnixJob {self.command}, super={super().__repr__()})"
    
    def __str__(self):
        return f"UnixJob {self.command}, super={super().__str__()})"
    
@dataclass
class UnixScheduler(Scheduler):
    """convenience class for local unix execution"""
    jobClazz:Type[Job] = UnixJob
    
    def __post_init__(self):
        super().__post_init__()
        # print(self.get_save_attributes())
        pass
    
    def get_save_attributes(self):
        return super().get_save_attributes()
    
    def __repr__(self):
        return super().__repr__()
    
    def __str__(self):
        return super().__str__()