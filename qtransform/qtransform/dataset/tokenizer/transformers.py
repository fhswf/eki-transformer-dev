from typing import List
from tiktoken import get_encoding, Encoding
from qtransform.dataset.tokenizer import Tokenizer, get_files, save_tokens
from omegaconf import DictConfig
import logging
from numpy import array, dtype

log = logging.getLogger(__name__)

class TransformersTokenizer(Tokenizer):
    """
        Uses transformer as the tokenizer of choice.
    """
    def tokenize(tokenizer_cfg: DictConfig):
        """
            Tokenize an input from file under text_file and write the generated bin file in root_path/root_path-<encoding>.bin. 
            Since BPE is used, it is possible that the value of tokens cannot be supported by the datatype due to insufficient amount of bits.
        """
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        #TODO: transformers tokenization
        ids = array(tokens, dtype=tokenizer_cfg.dtype)
        save_tokens(ids, tokenizer_cfg)

