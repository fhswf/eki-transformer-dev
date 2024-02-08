import numpy as np
from qtransform.classloader import get_data
from .tokenizer import Tokenizer, Metadata
import logging 
from omegaconf import DictConfig
from pprint import PrettyPrinter

log = logging.getLogger(__name__)

from .character import CharacterTokenizer
from .tiktoken import TikTokenizer
from .transformers import TransformersTokenizer

__all__ = ['CharacterTokenizer', 'TikTokenizer', 'TransformersTokenizer']

import qtransform.dataset.tokenizer as package_self

def get_tokenizer(tokenizer_cfg: DictConfig, memmap: np.memmap = None) -> Tokenizer:
    """
        Tokenizes a text based on the hydra configuration. It encodes a text based on the encoding property and saves the output with 
        the datatype dtype in a numpy array binary file. For some tokenizers like character encoding, the encoding property is ignored.
    """
    ## work around for issue https://github.com/openai/tiktoken/pull/181 and https://github.com/huggingface/datasets/issues/5536
    import copyreg, functools, tiktoken
    def pickle_Encoding(enc):
        return (functools.partial(tiktoken.core.Encoding, enc.name, pat_str=enc._pat_str, mergeable_ranks=enc._mergeable_ranks, special_tokens=enc._special_tokens), ())
    copyreg.pickle(tiktoken.core.Encoding, pickle_Encoding)
    log.debug(f'Attempting to retrieve tokenizer with cfg: {PrettyPrinter(indent=1).pformat(tokenizer_cfg)}')
    tokenizer: Tokenizer = get_data(log, package_self, tokenizer_cfg.wrapper, Tokenizer, args={"tokenizer_cfg": tokenizer_cfg, "memmap": memmap})
    return tokenizer