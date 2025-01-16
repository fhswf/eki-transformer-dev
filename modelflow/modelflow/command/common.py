import abc
import os
import subprocess
import logging
from dataclasses import dataclass
from typing import Optional, TypeVar
from modelflow.utils.helper import singleton
from modelflow.command.extract import Extractor
log = logging.getLogger(__name__)

E = TypeVar("E", bound=Extractor)
@dataclass
class Command(abc.ABC):
    cmd_args: str
    extractor: Optional[E]
    @abc.abstractmethod
    def run(self, *args, **kwargs):
        raise NotImplementedError
        pass

@dataclass
class SystemCommand(Command):
    cmd_bin: str
    
    def subprocess_run(self, shell=bool):
        result = None
        try:
            result: subprocess.CompletedProcess = subprocess.run(self.cmd_bin + " " + self.cmd_args, shell=shell, check=True)    
        except subprocess.CalledProcessError as e:
            log.exception(f"subprocess ended with error {e}")
        return result

    def run(self):
        return self.subprocess_run(True)

@singleton
@dataclass
class CommandCounter():
    """count executed commands and store command ids in a dict"""
    counter:int = 0
    command_dict:dict = {}

    def inc(self, cmd=None):
        if cmd is None:
            self.command_dict.update({self.counter: cmd})
        self.counter = self.counter + 1
        return self.counter
    
    def get_counter(self):
        return self.counter
    
    def get_command_dict(self):
        return self.command_dict