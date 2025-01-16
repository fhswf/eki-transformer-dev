from modelflow.scheduler.common import Scheduler
import re
import time
import logging
import subprocess 
log = logging.getLogger(__name__)

class UnixScheduler(Scheduler):
    def __init__(self, cfg):
        super().__init__()
        # list if paths to jobsfiles on the drive
        self.jobfiles = [] 

    def create_job(self):
        pass

    def run_jobs(self, commands):
        """Runs the given list of commands (as Slurm jobs) according to the scheduler's policy."""
        for job in self.jobfiles:
            job_id = self.submit_command(job)
            if job_id:
                # Wait for the job to complete
                log.info(f"Waiting for job {job_id} to complete...")
                while not self.is_job_complete(job_id):
                    time.sleep(10)  # Check every 10 seconds
                log.info(f"Job {job_id} completed")
                
    def run(self):
        self.run_jobs(self.commands)
