from dataclasses import dataclass
from typing import Any, Dict, List
from omegaconf import DictConfig
from qtransform.classloader import get_data
import logging
from qtransform.utils.introspection import concat_paths
from abc import ABC, abstractclassmethod
from numpy import dtype
from glob import glob
from os.path import isdir, exists
import os
from numpy import ndarray
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

def get_files(tokenizer_cfg: DictConfig) -> List:
    """
        Returns all readable files from a given directory. Currently, only files at root level are returned.
    """
    root_path: list = tokenizer_cfg.dataset_dir #append directory seperator at the end of the path
    raw_dir = os.path.join(root_path, "untokenized", "")
    # raw_dir = concat_paths(raw_dir)
    if not exists(raw_dir):
        log.debug(f'Creating directory {raw_dir}')
        os.makedirs(raw_dir, exist_ok=True)
    log.debug(f'Checking for files with name containing {tokenizer_cfg.name} under directory: {raw_dir}')
    return [x for x in glob(raw_dir + tokenizer_cfg.name + '*') if not isdir(x)]

def save_tokens(ids: ndarray,tokenizer_cfg: DictConfig, meta: Dict = None) -> None:
    """
        Saves the tokens from an ndarray into a binary file. If meta is passed, a file containing metadata about
        the tokens is created.
    """
    root_path: list = tokenizer_cfg.dataset_dir
    output_dir = os.path.join(root_path, "tokenized", "")
    filename = tokenizer_cfg.name + "-" + tokenizer_cfg.encoding + "-" + tokenizer_cfg.dtype
    #directory seperator included in output_dir
    if not exists(output_dir):
        log.debug(f'Creating directory {output_dir}')
        os.makedirs(output_dir, exist_ok=True)
    if meta != None:
        with open(output_dir + filename + '-meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
    #write numpy array to a binary file
    ids.tofile(output_dir + filename + ".bin")