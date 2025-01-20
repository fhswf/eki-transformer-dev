from modelflow.command.common import MetaTaskList
from dataclasses import dataclass
import logging
log = logging.getLogger(__name__)

@dataclass
class Auto(MetaTaskList):
    pass

@dataclass    
class LayerExchange(Auto):
    source_layer: str
    target_layer: str
    strategy: str # TODO replace this with enum or class
    until: str  # TODO replace this with enum or class

    def __post_init__(self):
        # prepare call pipeline, do we replace layer by layer here or within qtransform?
        # logic here could be done on a graph basis independend on training framework
        self.tasks = {}
        raise NotImplementedError
        
    #def run(self, *args, **kwargs):
    #    super()