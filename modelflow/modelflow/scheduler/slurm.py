from modelflow.scheduler.common import Scheduler
import re
import time
import logging
import subprocess 
log = logging.getLogger(__name__)


class SlurmScheduler(Scheduler):
    def __init__(self, cfg):
        super().__init__()
        # list if paths to jobsfiles on the drive
        self.jobfiles = [] 

    def create_slurm_jobfile(self):
        """
        create a slurm job based on some template for each job that this sheduler is supposed to run.
        returns a list of ordered temporary job files.
        """
        return None
    
    def create_job(self):
        pass

    def submit_command(self, command):
        """
        Submits a command using sbatch and returns the job ID.
        command has to be a slurm job description
        """
        # Submit the job and capture the output
        result = subprocess.run(['sbatch', command], shell=True, capture_output=True, text=True)
        
        # Extract the job ID from the output
        match = re.search(r'Submitted batch job (\d+)', result.stdout)
        if match:
            job_id = match.group(1)
            log.info(f"Submitted job {job_id}")
            return job_id
        else:
            log.error("Possibly failed to submit job")
            log.info("process out: TODO")
            raise

    def is_job_complete(self, job_id):
        """Checks if the given job ID has completed."""
        result = subprocess.run(['squeue', '-j', job_id, '--noheader'], capture_output=True, text=True)
        return result.stdout.strip() == ""

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
                
# Example usage
if __name__ == "__main__":
    scheduler = Scheduler()
    commands = ['/path/to/your_script1.sh', '/path/to/your_script2.sh']  # Replace these with your actual script paths
    scheduler.run_jobs(commands)