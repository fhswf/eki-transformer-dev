from tiktoken import get_encoding
from typing import Callable
from qtransform.dataset.tokenizer import Tokenizer
import os
from omegaconf import DictConfig
import logging

log = logging.getLogger(__name__)

class TikTokenizer(Tokenizer):
    """
        Uses tiktoken as the tokenizer of choice.
    """
    def __init__():
        pass
    
    def get_tokenizer(cfg: DictConfig):
        pass

    def set_encoding(self,encoding: str):
        self.encoding = encoding

    def encode(text_file: str ,root_path: str):
        "Tokenize an input from file under text_file and write the generated bin file in root_path/root_path-<encoding>.bin"
        pass