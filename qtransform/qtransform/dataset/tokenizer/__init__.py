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
    max_token_value: int

    @abstractclassmethod
    def tokenize(memmap: np.memmap, text: str, tokenizer_cfg: DictConfig):
        """
            Tokenize a text and write the result into a memmap to be retrieved later. 
            To differentiate between datasets which contain multiple or single samples per row, 
            the shape of the memmap is taken into consideration. 
        """

    def get_memmap_dimension(memmap: np.memmap) -> Tuple[int]:
        num_items: int = memmap.shape[0]
        if len(memmap.shape) == 1: #entire text is one large 1d array
            num_rows: int = 1
        else:
            num_rows: int = memmap_shape[1]
        return (num_items, num_rows)
    
    @abstractclassmethod
    def save_metadata():
        pass


import qtransform.dataset.tokenizer as package_self

def get_tokenizer(tokenizer_cfg: DictConfig) -> Tokenizer:
    """
        Tokenizes a text based on the hydra configuration. It encodes a text based on the encoding property and saves the output with 
        the datatype dtype in a numpy array binary file. For some tokenizers like character encoding, the encoding property is ignored.
    """
    tokenizer: Tokenizer = get_data(log, package_self, tokenizer_cfg.wrapper, Tokenizer)
    return tokenizer

def save_tokens(ids: np.ndarray, tokenizer_cfg: DictConfig, meta: Dict = None) -> None:
    """
        Saves the tokens from an ndarray into a binary file. If meta is passed, a file containing metadata about
        the tokens is created.
    """
    output_dir = concat_paths([*tokenizer_cfg.dataset_dir, "tokenized", tokenizer_cfg.encoding, ""])
    filename = tokenizer_cfg.name + "-" + tokenizer_cfg.encoding + "-" + tokenizer_cfg.dtype
    #directory seperator included in output_dir
    if not exists(output_dir):
        log.debug(f'Creating directory {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
    if meta != None:
        with open(output_dir + filename + '-meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
    #write numpy array to a binary file
    #this is inefficient for larger datasets as the entire array has to be stored within memory to be saved
    #instead, use memmap with write permissions
    #ids.tofile(output_dir + filename + ".bin")