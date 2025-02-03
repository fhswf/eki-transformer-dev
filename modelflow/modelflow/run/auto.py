from modelflow.command.common import Task, TaskInterator
from modelflow.scheduler.common import Serializable
from dataclasses import dataclass
from typing import Any
import logging
log = logging.getLogger(__name__)


@dataclass    
class LayerExchange(TaskInterator, Serializable):
    # TODO only works for qtransform atm 
    cmd_args : str = ""
    source_layer: str = ""
    target_layer: str = ""
    strategy: Any = None # TODO replace this with enum or class
    until: Any  = None # TODO replace this with enum or class
    current_step: Any = None
    def __post_init__(self):
        # prepare call pipeline, do we replace layer by layer here or within qtransform?
        # logic here could be done on a graph basis independend on training framework
        # self.tasks = [] # done in super
        pass
    
    def create_sub_task(self)->Task:
        """create a sub tasks for a conversion step and fill missing parts of sub tasks with task content of this taskiterators common attr"""
        pass
    
    def get_save_attributes(self):
        return ["source_layer", "target_layer", "strategy", "until", "current_step"]

    def __after_task__(self):
        # check if conversion was completed by checking the saved model, or the output logs
        pass