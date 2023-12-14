from dataclasses import dataclass
from typing import Dict, List, Tuple
from omegaconf import DictConfig
from qtransform.classloader import get_data
import logging
from qtransform.utils.introspection import concat_paths
from abc import ABC, abstractclassmethod
from numpy import dtype
from glob import glob
from os.path import isdir, exists
import os
import numpy as np
import pickle
#TODO: maybe implement pytorch.utils.get_tokenizer()

log = logging.getLogger(__name__)

@dataclass
class Metadata():
    """
        Defines the structure of pickled metadata files.
    """
    vocab_size: int
    encoding: str
    dtype: str

class Tokenizer(ABC):
    """
        Capsule around different implementations of tokenizers, to unify their interfaces.
        Each TokenizerWrapper has to contain a method to tokenize the data according to an encoding and store it in a numpy array of np.dtype
        on the harddrive. The necessary configuration parameters are included in the tokenizer_config method parameter
    """
    max_token_value: int
    vocab_size: int
    num_tokens: int

    def __init__(self, memmap: np.memmap, tokenizer_cfg: DictConfig):
        if not isinstance(memmap, np.memmap):
            log.error(f'Wrong type for memmap during tokenization ({memmap}, {type(memmap)})')
            raise TypeError()
        if len(memmap.shape) > 1 and memmap.shape[1] != 1:
            log.error(f'The memmap needs to be one dimensional during tokenization')
            raise ValueError()
        if memmap.mode != 'w+':
            log.error(f'Mode of memmap needs to be "w+" for tokenization.')
            raise AttributeError()
        if not isinstance(tokenizer_cfg, DictConfig):
            log.error(f'Tokenizer config is not a DictConfig ({tokenizer_cfg})')
            raise TypeError()
        self.memmap = memmap
        self.tokenizer_cfg = tokenizer_cfg
        self.vocab_size = 0
        self.num_tokens = 0

    @abstractclassmethod
    def tokenize(self, text: str):
        """
            Tokenize a text and write the result into a memmap to be retrieved later. 
            The memmap is expected to be a 1d array in which the tokenized text is written continuously.
            If it is not one dimensional, an error will be thrown.
        """

    @abstractclassmethod
    def save_metadata(self, filepath: str):
        pass


import qtransform.dataset.tokenizer as package_self

def get_tokenizer(tokenizer_cfg: DictConfig, memmap: np.memmap) -> Tokenizer:
    """
        Tokenizes a text based on the hydra configuration. It encodes a text based on the encoding property and saves the output with 
        the datatype dtype in a numpy array binary file. For some tokenizers like character encoding, the encoding property is ignored.
    """
    tokenizer: Tokenizer = get_data(log, package_self, tokenizer_cfg.wrapper, Tokenizer, args={"tokenizer_cfg": tokenizer_cfg, "memmap": memmap})
    return tokenizer