from modelflow.command.common import Task
from modelflow.scheduler.common import Serializable
from dataclasses import dataclass
import logging
log = logging.getLogger(__name__)


@dataclass    
class LayerExchange(Task, Serializable):
    cmd_args : str = None
    source_layer: str = None
    target_layer: str = None
    strategy: str = None # TODO replace this with enum or class
    until: str  = None # TODO replace this with enum or class
    current_step = None
    def __post_init__(self):
        # prepare call pipeline, do we replace layer by layer here or within qtransform?
        # logic here could be done on a graph basis independend on training framework
        self.tasks = {}
        raise NotImplementedError

    def get_save_attributes(self):
        return ["source_layer", "target_layer", "strategy", "until", "current_step"]
        
    def run(self, *args, **kwargs):
        pass