from dataclasses import dataclass
import abc
import logging
from modelflow.command.common import Command
from typing import List, Union, TypeVar

log = logging.getLogger(__name__)
    
S = TypeVar('S', bound='TaskList')
C = TypeVar('C', bound='Command')

@dataclass
class TaskList(abc.ABC):
    tasks: List[Union[S, C]]
    
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        pass

@dataclass
class MetaTaskList(TaskList):
    common_cmd_args: str


@dataclass
class Sequence(MetaTaskList):
    
    def run(self):
        raise NotImplementedError
    
@dataclass
class Parrallel(MetaTaskList):
    
    def run(self):
        raise NotImplementedError