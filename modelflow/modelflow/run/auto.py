from modelflow.run.common import TaskList
from dataclasses import dataclass
import logging
log = logging.getLogger(__name__)

@dataclass
class Auto(TaskList):
    
    def run(self):
        pass