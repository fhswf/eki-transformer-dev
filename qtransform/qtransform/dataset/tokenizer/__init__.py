from dataclasses import dataclass
from typing import Dict, List, Tuple
from omegaconf import DictConfig, OmegaConf
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
from pprint import PrettyPrinter
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
        Generic wrapper around different tokenizer implementations to unify their interfaces within the project. 
        The tokenizer is stateful, storing the number of currently tokenized tokens as well as an optional memmap
        object to be used to write the tokens onto the harddrive. Alternatively, one could use the tokenize()
        method to retrieve a list of tokens from a text.
    """
    max_token_value: int
    num_tokens: int #the amount of tokens which have been encoded
    _memmap: np.memmap

    def __init__(self, tokenizer_cfg: DictConfig, memmap: np.memmap = None):
        if isinstance(tokenizer_cfg, Dict):
            log.debug(f'Tokenizer config is of type dict. Creating DictConfig object.')
            self.tokenizer_cfg = OmegaConf.create(tokenizer_cfg)
        elif not isinstance(tokenizer_cfg, DictConfig):
            log.error(f'Tokenizer config is not a DictConfig ({tokenizer_cfg})')
            raise TypeError()
        log.debug(f'Creating Tokenizer with parameters: {tokenizer_cfg}')
        self.tokenizer_cfg = tokenizer_cfg
        self.max_token_value = 0
        self.num_tokens = 0
        #tokenization can use memmap directly or simply return a list of integers
        if memmap is not None:
            self.memmap = memmap

    @property
    def memmap(self):
        return self._memmap

    @memmap.setter
    def memmap(self, value: np.memmap):
        if not isinstance(value, np.memmap):
            log.error(f'Wrong type for memmap during tokenization ({value}, {type(value)})')
            raise TypeError()
        if len(value.shape) > 1 and memmap.shape[1] != 1:
            log.error(f'The memmap needs to be one dimensional during tokenization')
            raise ValueError()
        if value.mode != 'w+':
            log.error(f'Mode of memmap needs to be "w+" for tokenization.')
            raise AttributeError()
        self._memmap = value

    @abstractclassmethod
    def tokenize_memmap(self, text: str):
        """
            Tokenize a text and write the result into a memmap to be retrieved later. 
            The memmap is expected to be a 1d array in which the tokenized text is written continuously.
            If it is not one dimensional, an error will be thrown.
        """
        if not isinstance(text, str):
            log.error(f'Text to tokenize is not a string')
            raise TypeError()
        if self.memmap is None:
            log.error(f'Memmap was not set')
            raise TypeError()

    @abstractclassmethod
    def tokenize(self, text: str) -> List[int]:
        """
            Tokenize a text and return the tokens in form of a list of integers.
            Unlike tokenize_memmap, the tokens are not written into a memmap file. 
        """
        if not isinstance(text, str):
            log.error(f'Text to tokenize is not a string')
            raise TypeError()

    @abstractclassmethod
    def decode(self, idx: List[int]) -> str:
        """
            Decodes a list of tokens into a sentence.
        """
        pass

    @abstractclassmethod
    def save_metadata(self, filepath: str):
        pass

    def _save_metadata(self, filepath, meta: Dict):
        """
            Saves metadata of a tokenized file from a meta object into filepath. This should include the encoding and information 
            about the vocabulary.
        """
        if os.path.isfile(filepath):
            directory, file_name = os.path.split(filepath)
        else:
            directory = filepath
            file_name = 'meta.pkl'
        if not exists(directory):
            log.debug(f'Creating directory {directory}')
            os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, file_name)
        with open(path, 'wb') as f:
            pickle.dump(meta, f)

    def check_dtype_overflow():
        if len(self.max_token_value) > 2 ** self.memmap.dtype.itemsize * 8 -1:
            log.error(f'Vocab size is larger than what the dtype can store ({self.memmap.dtype})')
            raise TypeError() 

import qtransform.dataset.tokenizer as package_self

def get_tokenizer(tokenizer_cfg: DictConfig, memmap: np.memmap = None) -> Tokenizer:
    """
        Tokenizes a text based on the hydra configuration. It encodes a text based on the encoding property and saves the output with 
        the datatype dtype in a numpy array binary file. For some tokenizers like character encoding, the encoding property is ignored.
    """
    log.debug(f'Attempting to retrieve tokenizer with cfg: {PrettyPrinter(indent=1).pformat(tokenizer_cfg)}')
    tokenizer: Tokenizer = get_data(log, package_self, tokenizer_cfg.wrapper, Tokenizer, args={"tokenizer_cfg": tokenizer_cfg, "memmap": memmap})
    return tokenizer