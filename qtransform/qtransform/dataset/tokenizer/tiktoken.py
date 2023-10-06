from typing import List
from tiktoken import get_encoding, Encoding
from qtransform.dataset.tokenizer import Tokenizer, get_files, save_tokens
from omegaconf import DictConfig
import logging
from numpy import array

log = logging.getLogger(__name__)

"""

  tokenizer:
    dataset_dir:
    - ${dataset.root_path}
    - data
    - ${dataset.name}
    name: ${dataset.name}
    dtype: ${data.dtype}
    wrapper: CharacterTokenizer
    encoding: character
"""

class TikTokenizer(Tokenizer):
    """
        Uses tiktoken as the tokenizer of choice. It calls encode_ordinary with an object
        containing the specified encoding, e.g. gpt2.
    """
    def tokenize(tokenizer_cfg: DictConfig):
        "Tokenize an input from file under text_file and write the generated bin file in root_path/root_path-<encoding>.bin"
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        encoder: Encoding = get_encoding(tokenizer_cfg.encoding)
        tokens: List[int] = list()
        for file in get_files(tokenizer_cfg):
            with open(file, 'r') as f:
                try:
                    data = f.read()
                    data_ids = encoder.encode_ordinary(data)
                    log.debug(f"Amount of tokens in dataset {file}: {len(data_ids)}")
                    tokens.extend(data_ids)
                except PermissionError:
                    pass
        ids = array(tokens, dtype=tokenizer_cfg.dtype)
        save_tokens(ids, tokenizer_cfg)