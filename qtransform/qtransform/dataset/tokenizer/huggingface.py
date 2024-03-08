from typing import List, Dict, Union
from omegaconf import DictConfig
import logging

from qtransform.dataset.tokenizer.tokenizer import Tokenizer2
log = logging.getLogger(__name__)

from numpy import array, dtype
from dataclasses import dataclass, asdict

from transformers import AutoTokenizer

class HuggingfaceTokenizer(Tokenizer2):
    """
        Currently not integrated with the rest....
        Uses unmodifed HUgginface Auto tokenizer.
    """
    def __init__(self, cfg:DictConfig):
        super().__init__(cfg)
        print(cfg)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pass
    
    def _encode(self, text) -> List[int]:
        return self.tokenizer.__len__encoder.encode(text)

    def _decode(self, idx: List[int]) -> str:
        return self.tokenizer.decoder.decode(idx)
    
    # delegations to hf tokenizer
    def tokenize(self, *args, **kwargs):
        self.tokenizer.tokenize(args, kwargs)

    def __len__(self, *args, **kwargs):
        self.tokenizer.__len__( *args, **kwargs)