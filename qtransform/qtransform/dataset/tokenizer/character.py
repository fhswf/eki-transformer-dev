from numpy import array, memmap
from omegaconf import DictConfig
from qtransform.dataset.tokenizer import Tokenizer, Metadata
import logging
from typing import List, Dict
import pickle
from os.path import join, exists, isfile
from dataclasses import dataclass, asdict

log = logging.getLogger(__name__)

#for now, metadata is only used for character tokenization
@dataclass
class CharacterMetadata(Metadata):
    itos: Dict[int, str] = None
    stoi: Dict[str, int] = None

    def __post_init__(self):
        if not isinstance(self.itos, Dict):
            self.itos = {0: "<UNKNOWN>"}
        if not isinstance(self.stoi, Dict):
            self.stoi = {"<UNKNOWN>": 0}

class CharacterTokenizer(Tokenizer):

    """
        Tokenizes one or multiple files inside of a directory into a singular binary file containing the numerical representation of
        each character accross all files. The id of each character is assigned iteratively (starting from one) and put into 
        a numpy array, with the first id being reserved for unknown characters.
        The data type is specified by the hydra config under data.dtype. It also creates a metadata file (.pkl) containing the
        vocabulary size and the corresponding mapping of characters to ids.
        The tokenization part is inspired by nanoGPT (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py)
    """

    def __init__(self, tokenizer_cfg, memmap: memmap = None):
        super().__init__(tokenizer_cfg, memmap)
        #python chars are at max 4B large, meaning that at the absolute most the vocab size is 2^32 (4 million)
        #that never happens as every single unicode character would need to appear within dataset
        self.meta: CharacterMetadata = CharacterMetadata(**asdict(self.meta))

    def decode(self, l: List[int]) -> str:
        return ''.join([self.meta.itos[i] if i in self.meta.itos else self.meta.itos[0] for i in l]) # decoder: take a list of integers, output a string
    
    def tokenize_memmap(self, text: str):
        #log.debug(f'Tokenizing text:\n"{text}"\nWith parameters: {self.tokenizer_cfg}')
        super().tokenize_memmap(text) #arg checking
        #self.check_dtype_overflow()
        self.memmap[self.meta.num_tokens : self.meta.num_tokens + len(text)] = self.encode(text)

    def encode(self, text: str) -> List[int]:
        #only update vocab with new characters
        new_tokens = set(text) - set(self.meta.stoi.keys()) 
        #remember the token of previous characters, start counting at max_token_value
        #merge encoding and training together as vocab is expanded during encoding process
        self.meta.stoi.update({ ch:i for i,ch in enumerate(new_tokens, self.meta.max_token_value + 1) })
        self.meta.itos.update({ i:ch for i,ch in enumerate(new_tokens, self.meta.max_token_value + 1) })
        self.meta.max_token_value = len(self.meta.itos.keys()) - 1 #tokens start at 0
        self.meta.num_tokens += len(text)
        #unknown tokens are saved in the vocabulary during encoding
        return [self.meta.stoi[c] for c in text]