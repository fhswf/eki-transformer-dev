from typing import List, Dict
from tiktoken import get_encoding, Encoding
from qtransform.dataset.tokenizer import Tokenizer, Metadata
from omegaconf import DictConfig
import logging
from numpy import array, dtype
from dataclasses import dataclass, asdict

log = logging.getLogger(__name__)

@dataclass
class TiktokenMetadata(Metadata):
    module: str = "tiktoken"

class TikTokenizer(Tokenizer):
    """
        Uses tiktoken as the tokenizer of choice. It calls encode_ordinary with an object
        containing the specified encoding, e.g. gpt2.
    """
    def __init__(self, tokenizer_cfg, memmap = None):
        super().__init__(tokenizer_cfg = tokenizer_cfg, memmap=memmap)
        self.meta: TiktokenMetadata = TiktokenMetadata(**asdict(self.meta))
        try:
            self.encoder: Encoding = get_encoding(self.meta.encoding)
        except ValueError as e:
            log.error(f'Could not load Tiktoken tokenizer with encoding: "{self.meta.encoding}".')
            raise ValueError()
        self.meta.max_token_value = self.encoder.max_token_value

    def encode(self, text) -> List[int]:
        tokens = self.encoder.encode_ordinary(text)
        self.meta.num_tokens += len(tokens) #only relevant for memmap indexing
        return tokens

    def decode(self, idx: List[int]) -> str:
        return self.encoder.decode(idx)
    
    def get_pad_token(self):
        return "<|endoftext|>"
    
    def get_pad_id(self):
        return self.encoder.max_token_value