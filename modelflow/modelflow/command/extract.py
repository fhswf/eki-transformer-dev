from dataclasses import dataclass
from abc import ABC, abstractmethod
from modelflow.command.common import CommandCounter

@dataclass
class Extractor(ABC):
    """ Base class to extract some result from a command or execution"""
    id: str = None # id for retrival
    
    def __post_init__(self):
        if self.id is None:
            # generate id for this commands output
            pass
        pass
    
    @abstractmethod
    def run(self):
        pass 

@dataclass
class StdoutStrExtractor(Extractor):
    regex: str
    pass

@dataclass
class FileStrExtractor(Extractor):
    """read string from a file"""
    file_name: str
    regex: str
    pass

@dataclass
class FileCopyExtractor(Extractor):
    """copy a file"""
    file_name: str
    pass