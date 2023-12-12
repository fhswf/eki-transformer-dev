from typing import List
from qtransform.dataset.tokenizer import Tokenizer, get_files, save_tokens
from omegaconf import DictConfig
import logging
from numpy import array, dtype
import transformers
from datasets import Dataset


log = logging.getLogger(__name__)

class TransformersTokenizer(Tokenizer):
    """
        Uses transformer as the tokenizer of choice.
    """
    def tokenize(tokenizer_cfg: DictConfig, hf_dataset: Dataset):
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        #TODO: transformers tokenization
        ids = array(tokens, dtype=tokenizer_cfg.dtype)
        save_tokens(ids, tokenizer_cfg)

