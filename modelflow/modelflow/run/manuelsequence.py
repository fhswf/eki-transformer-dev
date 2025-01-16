from modelflow.run.common import TaskList
from dataclasses import dataclass
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
from modelflow import CFG
from modelflow.scheduler.common import Scheduler
from typing import TypeVar
import logging
log = logging.getLogger(__name__)

S = TypeVar("S", bound="Scheduler")

@dataclass
class ManuelSequence(TaskList):
    scheduler: Scheduler = instantiate(CFG().scheduler)
    def __post_init__(self):
        # self.scheduler = instantiate(CFG().scheduler)
        pass
    
    def run(self):
        pass