from dataclasses import dataclass
from typing import Any
from omegaconf import DictConfig
from qtransform.classloader import get_data
import logging
from qtransform.utils.introspection import get_classes
from abc import ABC, abstractclassmethod
from numpy import dtype
#TODO: maybe implement pytorch.utils.get_tokenizer()

log = logging.getLogger(__name__)


@dataclass
class TokenizerInfo:
    wrapper: str
    encoding: str
    dtype: dtype

class Tokenizer(ABC):
    """
        Capsule around different implementations of tokenizers, to unify their interfaces.
        Each TokenizerWrapper has to contain a method to tokenize the data according to an encoding and store it in a numpy array of np.dtype
        on the harddrive. The necessary configuration parameters are included in the tokenizer_config method parameter
    """
    @abstractclassmethod
    def tokenize(tokenizer_cfg: DictConfig):
        "Tokenize an input from file under text_file and write the generated bin file in root_path/root_path-<encoding>.bin"
        pass


import qtransform.dataset.tokenizer as package_self

def get_tokenizer(tokenizer_cfg: DictConfig) -> Tokenizer:
    """
        Tokenizes a text based on the hydra configuration. It encodes a text based on the encoding property and saves the output with 
        the datatype dtype in a numpy array binary file. For some tokenizers like character encoding, the encoding property is ignored.
    """
    tokenizer: Tokenizer = get_data(log, package_self, tokenizer_cfg.wrapper, Tokenizer)
    return tokenizer

def encode(tokenizer_cfg: DictConfig) -> None:
    """
        Basically does the same as get_tokenizer except that instead of returning an Object/ Class of Tokenizer, the tokenize method
        is called
    """
    tokenizer: Tokenizer = get_data(log, package_self, tokenizer_cfg.wrapper, Tokenizer)
    tokenizer.tokenize(tokenizer_cfg)