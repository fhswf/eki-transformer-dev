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
    def __init__(self, memmap, tokenizer_cfg):
        super().__init__(memmap, tokenizer_cfg)
        try:
            self.encoder: Encoding = get_encoding(tokenizer_cfg.encoding)
        except ValueError as e:
            log.error(f'Could not load Tiktoken tokenizer with encoding: "{tokenizer_cfg.encoding}".')
            raise ValueError()
        self.max_token_value = self.encoder.max_token_value

    def tokenize(self, text: str):
        """
            Tokenize an input from file under text_file and write the generated bin file in root_path/root_path-<encoding>.bin. 
            Since BPE is used, it is possible that the value of tokens cannot be supported by the datatype due to insufficient amount of bits.
        """
        tokens = self.encoder.encode_ordinary(text)
        #self.check_dtype_overflow()
        self.memmap[self.num_tokens: self.num_tokens + len(tokens)] = tokens
        self.num_tokens += len(tokens)

    def save_metadata(self, filepath):
        meta = {
            'max_token_value': self.max_token_value,
            'encoding': self.tokenizer_cfg.encoding,
            'module': 'tiktoken',
            'dtype': self.tokenizer_cfg.dtype
        }
        self._save_metadata(filepath, meta)