from tiktoken import get_encoding
from typing import Callable
from qtransform.dataset.tokenizer import Tokenizer
import os
from omegaconf import DictConfig
import logging
import numpy as np

log = logging.getLogger(__name__)

class TikTokenizer(Tokenizer):
    """
        Uses tiktoken as the tokenizer of choice.
    """
    #root_path: str, text_file: str, encoding: str, dtype: np.dtype
    def encode(tokenize_cfg: DictConfig):
        pass
        raise NotImplementedError("not implemented yet")
        "Tokenize an input from file under text_file and write the generated bin file in root_path/root_path-<encoding>.bin"
        encoder = get_encoding(tokenize_cfg.encoding)
        with open(text_file, 'r') as file:
            data = file.read()
        data_ids = encoder.encode_ordinary(data)
        data_ids_np = np.array(data_ids, dtype=dtype)
        data_ids_np.tofile(root_path + "tokenized")