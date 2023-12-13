from typing import List
from qtransform.dataset.tokenizer import Tokenizer
from omegaconf import DictConfig
import logging
import transformers
from datasets import Dataset#, PreTrainedTokenizer, PreTrainedTokenizerFast
from qtransform.classloader import get_data

log = logging.getLogger(__name__)

class TransformersTokenizer(Tokenizer):
    """
        Uses transformers as the tokenizer of choice.
        You can choose between the "full" version or the fast version. Differences can be read on 
        https://huggingface.co/docs/transformers/main_classes/tokenizer#tokenizer (from 13.12.2023)
    """
    def tokenize(tokenizer_cfg: DictConfig, hf_dataset: Dataset):
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        #TODO: transformers tokenization
        ids = array(tokens, dtype=tokenizer_cfg.dtype)
        save_tokens(ids, tokenizer_cfg)

