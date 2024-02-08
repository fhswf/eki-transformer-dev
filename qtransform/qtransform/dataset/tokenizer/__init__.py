from dataclasses import dataclass, asdict, replace, fields
from typing import Dict, List, Tuple, Union, Any
from omegaconf import DictConfig, OmegaConf, open_dict
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
import inspect

log = logging.getLogger(__name__)

@dataclass
class Metadata():
    """
        Defines the structure of pickled metadata files. If a tokenizer needs additional fields, another dataclass deriving
        from metadata should be created and its signature used.
    """
    max_token_value: int
    encoding: str
    dtype: str
    num_tokens: int = 0
    module: str = ""

class Tokenizer(ABC):
    """
        Generic wrapper around different tokenizer implementations to unify their interfaces within the project. 
        The tokenizer is stateful, storing the number of currently tokenized tokens as well as an optional memmap
        object to be used to write the tokens onto the harddrive. Alternatively, one could use the tokenize()
        method to retrieve a list of tokens from a text.
    """
    _memmap: np.memmap
    _meta: Metadata

    def __init__(self, tokenizer_cfg: DictConfig, memmap: np.memmap = None):
        if isinstance(tokenizer_cfg, Dict):
            log.debug(f'Tokenizer config is of type dict. Creating DictConfig object.')
            self.tokenizer_cfg = OmegaConf.create(tokenizer_cfg)
        elif not isinstance(tokenizer_cfg, DictConfig):
            log.error(f'Tokenizer config is not a DictConfig ({tokenizer_cfg})')
            raise TypeError()
        log.debug(f'Creating Tokenizer with parameters: {tokenizer_cfg}')
        self.tokenizer_cfg = tokenizer_cfg
        self.meta = Metadata(max_token_value = 0, encoding = self.tokenizer_cfg.encoding, dtype=self.tokenizer_cfg.dtype, num_tokens=0)
        if self.tokenizer_cfg.get('meta_file') is None:
            with open_dict(self.tokenizer_cfg):
                log.warning(f'Property meta_file omited in config. Assuming default: "meta.pkl"')
                self.tokenizer_cfg["meta_file"] = "meta.pkl"
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
        if len(value.shape) > 1 and value.shape[0] != 1:
            log.error(f'The memmap needs to be one dimensional for tokenization')
            raise ValueError()
        if value.mode != 'w+':
            log.error(f'Mode of memmap needs to be "w+" for tokenization.')
            raise AttributeError()
        self._memmap = value

    @property
    def meta(self):
        return self._meta
    
    @meta.setter
    def meta(self, value: Metadata):
        if isinstance(value, Union[Dict, DictConfig]):
            value = Metadata(**value)
        elif not isinstance(value, Metadata):
            log.error(f'Cannot use metadata of type: {type(value)}')
        self._meta = value

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
        offset = self.meta.num_tokens
        tokens: List[int] = self.encode(text)
        self.memmap[offset: offset + len(tokens)] = tokens

    @abstractclassmethod
    def encode(self, text: str, infer: bool = False) -> List[int]:
        """
            Tokenize a text and return the tokens in form of a list of integers.
            Unlike tokenize_memmap, the tokens are not written into a memmap file. 
            Depending on whether infer is set to True or False, the vocabulary and the number of tokens
            processed within the tokenizer are updated.
            TODO: taking different actions depending on infer param could slow down tokenization process by a tiny margin
        """
        if not isinstance(text, str):
            log.error(f'Text to tokenize is not a string')
            raise TypeError()

    @abstractclassmethod
    def decode(self, idx: List[int]) -> str:
        """
            Decodes a list of tokens into a sentence.
        """
        if not isinstance(idx, list):
            log.error(f'idx is not a list')
            raise TypeError()

    def save_metadata(self, filepath: str):
        """
            Saves metadata of a tokenized file as well as information about the tokenizer into filepath. 
            The information includes, but is not limited to: encoding, tokenizer_module, max_token_value.
            If character tokenization is used, the vocabulary is saved within the metadata file as well.

            args:
                filepath: the filepath. if filepath is a directory, meta.pkl is appended to it.
        """
        #file structure check
        if os.path.isfile(filepath):
            directory, file_name = os.path.split(filepath)
        else:
            directory = filepath
            file_name = self.tokenizer_cfg.meta_file #by default: meta.pkl
        if not exists(directory):
            log.debug(f'Creating directory {directory}')
            os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, file_name)

        #write
        with open(path, 'wb') as f:
            pickle.dump(asdict(self.meta), f)

    def load_metadata(self, filepath: str = None, meta: Dict = None):
        """
        Load metadata from a dictionary or from a file.
        """
        #TODO: tokenizer config contains meta file, here it is ignored. maybe refactor
        if filepath is not None:
            meta = self._load_metadata(filepath)
            #keys not supported in metadata of tokenizer (e.g. fast encoding for tiktoken)
            unsupported_keys = set(meta.keys()) - set([x.name for x in fields(self.meta)])
            unsupported_keys_dict = {x:meta[x] for x in unsupported_keys}
            if len(unsupported_keys) > 0:
                log.warning(f'Metadata contains keys {unsupported_keys_dict}.'\
                    f'They are not supported in {self.tokenizer_cfg.module}. Removing them.')
            meta = {x: meta[x] for x in meta if x not in unsupported_keys}
            meta["module"] = self.tokenizer_cfg.module
            self.meta =  replace(self.meta, **meta)
        elif isinstance(meta, Union[Dict, DictConfig]):
            self.meta =  replace(self.meta, **meta)
        else:
            log.error(f'Neither filepath was supplied nor is the supplied meta object of type dict (was type: {type(meta)})')
            raise TypeError()

    def _load_metadata(self, filepath: str) -> Dict[Any, Any]:
        if not isinstance(filepath, str):
            log.error(f'Error loading metadata from file: "{filepath}" is not a string.')
            raise TypeError()
        if not exists(filepath):
            log.error(f'Metadata under path "{filepath}" does not exist.')
            raise ValueError()
        if not os.path.isfile(filepath):
            log.error(f'Error while loading metadata for class {self.__class__.__name__}: "{filepath}" is not a file')
            raise ValueError()
        with open(filepath, 'rb') as pkl_file:
            meta = pickle.load(pkl_file)
        return meta

    @DeprecationWarning
    def meta_as_dict(self) -> Dict[str, Any]:
        #does the same as dataclasses.asdict
        params = set(inspect.signature(Metadata.__init__).parameters.keys()) - set(['self'])
        return {x:getattr(self.meta, x) for x in params}

    def check_dtype_overflow():
        if len(self.meta.max_token_value) > 2 ** self.memmap.dtype.itemsize * 8 -1:
            log.error(f'Vocab size is larger than what the dtype can store ({self.memmap.dtype})')
            raise TypeError() 

import qtransform.dataset.tokenizer as package_self

def get_tokenizer(tokenizer_cfg: DictConfig, memmap: np.memmap = None) -> Tokenizer:
    """
        Tokenizes a text based on the hydra configuration. It encodes a text based on the encoding property and saves the output with 
        the datatype dtype in a numpy array binary file. For some tokenizers like character encoding, the encoding property is ignored.
    """
    ## work around for issue https://github.com/openai/tiktoken/pull/181 and https://github.com/huggingface/datasets/issues/5536
    import copyreg, functools, tiktoken
    def pickle_Encoding(enc):
        return (functools.partial(tiktoken.core.Encoding, enc.name, pat_str=enc._pat_str, mergeable_ranks=enc._mergeable_ranks, special_tokens=enc._special_tokens), ())
    copyreg.pickle(tiktoken.core.Encoding, pickle_Encoding)
    log.debug(f'Attempting to retrieve tokenizer with cfg: {PrettyPrinter(indent=1).pformat(tokenizer_cfg)}')
    tokenizer: Tokenizer = get_data(log, package_self, tokenizer_cfg.wrapper, Tokenizer, args={"tokenizer_cfg": tokenizer_cfg, "memmap": memmap})
    return tokenizer