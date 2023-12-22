from typing import List
from tiktoken import get_encoding, Encoding
from qtransform.dataset.tokenizer import Tokenizer
from omegaconf import DictConfig
import logging
from numpy import array, dtype

log = logging.getLogger(__name__)

class TikTokenizer(Tokenizer):
    """
        Uses tiktoken as the tokenizer of choice. It calls encode_ordinary with an object
        containing the specified encoding, e.g. gpt2.
    """
    def __init__(self, tokenizer_cfg, memmap = None):
        super().__init__(tokenizer_cfg = tokenizer_cfg, memmap=memmap)
        try:
            self.encoder: Encoding = get_encoding(tokenizer_cfg.encoding)
        except ValueError as e:
            log.error(f'Could not load Tiktoken tokenizer with encoding: "{tokenizer_cfg.encoding}".')
            raise ValueError()
        self.max_token_value = self.encoder.max_token_value

    def tokenize_memmap(self, text: str):
        tokens: List[int] = self.tokenize(text)
        #self.check_dtype_overflow()
        offset = self.num_tokens
        self.memmap[offset: offset + len(tokens)] = tokens

    def tokenize(self, text) -> List[int]:
        tokens = self.encoder.encode_ordinary(text)
        self.num_tokens += len(tokens)
        return tokens

    def decode(self, idx: List[int]) -> str:
        return self.tokenizer.decode(idx)
        
    def save_metadata(self, filepath):
        meta = {
            'max_token_value': self.max_token_value,
            'encoding': self.tokenizer_cfg.encoding,
            'module': 'tiktoken',
            'dtype': self.tokenizer_cfg.dtype
        }
        self._save_metadata(filepath, meta)