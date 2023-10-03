from typing import Any
from omegaconf import DictConfig
from qtransform.classloader import get_data
import logging
from qtransform.utils.introspection import get_classes
from abc import ABC, abstractclassmethod

#TODO: maybe implement pytorch.utils.get_tokenizer()

log = logging.getLogger(__name__)



class Tokenizer(ABC):
    encoding: str
    """
    Capsule around different implementations of tokenizers, to unify their interfaces.
    Each TokenizerWrapper has to contain a method to import the tokenizer with the corresponding encoding
    """
    @abstractclassmethod
    def get_tokenizer():
        pass

    @abstractclassmethod
    def set_encoding(encoding: str):
        pass
    
    @abstractclassmethod
    def encode(text_file: str ,root_path: str):
        "Tokenize an input from file under text_file and write the generated bin file in root_path/root_path-<encoding>.bin"
        pass

def get_tokenizer(tokenizer_cfg: DictConfig) -> Tokenizer:
    pass
    """
        Returns a generic wrapper that is capable of encoding text. In this case, no generic wrapper for tokenizers like 
        the Dataset class for Dataset exists
    """
    import qtransform.dataset as package_self
    dataset_wrapper: Tokenizer = get_data(log, package_self, tokenizer_cfg.name, Tokenizer)
    return dataset_wrapper.load_dataset(tokenizer_cfg)