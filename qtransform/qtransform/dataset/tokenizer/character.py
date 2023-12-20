from numpy import array, memmap
from omegaconf import DictConfig
from qtransform.dataset.tokenizer import Tokenizer
import logging
from typing import List
import pickle
from os.path import join, exists

log = logging.getLogger(__name__)

class CharacterTokenizer(Tokenizer):

    """
        Tokenizes one or multiple files inside of a directory into a singular binary file containing the numerical representation of
        each character accross all files. The id of each character is assigned iteratively (starting from zero) and put into 
        a numpy array. The data type is specified by the hydra config under data.dtype. It also creates a metadata file (.pkl) containing the
        vocabulary size and the corresponding mapping of characters to ids.
        The tokenization part is heavily inspired by nanoGPT (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py)
    """

    def __init__(self, tokenizer_cfg, memmap: memmap = None):
        super().__init__(tokenizer_cfg, memmap)
        #python chars are at max 4B large, meaning that at the absolute most the vocab size is 2^32 (4 million)
        #that never happens as every single unicode character would need to appear within dataset
        self.max_token_value = 2**16 -1
        self.stoi: dict = dict()
        self.itos: dict = dict()
        self.vocab = list()

    def encode(self, s) -> List[int]:
        return [self.stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(self, l) -> str:
        return ''.join([self.itos[i] for i in l]) # decoder: take a list of integers, output a string
    
    def tokenize_memmap(self, text: str):
        #log.debug(f'Tokenizing text:\n"{text}"\nWith parameters: {self.tokenizer_cfg}')
        super().tokenize_memmap(text) #arg checking
        #self.check_dtype_overflow()
        self.memmap[self.num_tokens : self.num_tokens + len(text)] = self.tokenize(text)

    def tokenize(self, text: str) -> List[int]:
        #only update vocab with new characters
        new_tokens = set(text) - set(self.vocab) 
        #remember the token of previous characters, start counting at max_token_value
        self.stoi.update({ ch:i for i,ch in enumerate(new_tokens, len(self.vocab)) })
        self.itos.update({ i:ch for i,ch in enumerate(new_tokens, len(self.vocab)) })
        self.vocab.extend(new_tokens)
        self.max_token_value += len(self.vocab)
        self.num_tokens += len(text)
        #log.debug(f'Tokenizing: {text}. num_tokens: {self.num_tokens}')
        return self.encode(text)

    def save_metadata(self, filepath: str):
        # save the meta information as well, to help us encode/decode later
        meta = {
            'max_token_value': self.max_token_value,
            'encoding': self.tokenizer_cfg.encoding,
            'dtype': self.tokenizer_cfg.dtype,
            'itos': self.itos,
            'stoi': self.stoi
        }
        self._save_metadata(filepath, meta)