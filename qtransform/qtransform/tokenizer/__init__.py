import numpy as np
from qtransform.classloader import get_data
import logging 
from omegaconf import DictConfig
from pprint import PrettyPrinter
from typing import Union, Dict, Any

log = logging.getLogger(__name__)


from .tokenizer import Tokenizer, Metadata
from .character import CharacterTokenizer
from .tiktoken import TikTokenizer
from .transformers import TransformersTokenizer

__all__ = ['CharacterTokenizer', 'TikTokenizer', 'TransformersTokenizer']

import qtransform.tokenizer as package_self

def get_tokenizer(tokenizer_cfg: Union[DictConfig, Dict], memmap: np.memmap = None) -> Tokenizer:
    """
        Tokenizes a text based on the hydra configuration. It encodes a text based on the encoding property and saves the output with 
        the datatype dtype in a numpy array binary file. For some tokenizers like character encoding, the encoding property is ignored.
    """
    if not isinstance(tokenizer_cfg, Union[DictConfig, Dict]):
        log.error(f'Invalid type for tokenizer_cfg: {type(tokenizer_cfg)}')
        raise TypeError()
    ## work around for issue https://github.com/openai/tiktoken/pull/181 and https://github.com/huggingface/datasets/issues/5536
    import copyreg, functools, tiktoken
    def pickle_Encoding(enc):
        return (functools.partial(tiktoken.core.Encoding, enc.name, pat_str=enc._pat_str, mergeable_ranks=enc._mergeable_ranks, special_tokens=enc._special_tokens), ())
    copyreg.pickle(tiktoken.core.Encoding, pickle_Encoding)
    log.debug(f'Attempting to retrieve tokenizer with cfg: {PrettyPrinter(indent=1).pformat(tokenizer_cfg)}')
    tokenizer: Tokenizer = get_data(log, package_self, tokenizer_cfg.get("wrapper"), Tokenizer, args={"tokenizer_cfg": tokenizer_cfg, "memmap": memmap})
    return tokenizer

#def get_tokenizer2(cls_name: str, cfg:Union[DictConfig, Dict]) -> Any:
#    """ get tokenizer and return an instance """
#    log.debug(f" get tokenizer with config: {cfg}")
#    return load_class(logger=log, module=package_self, parent_class=Tokenizer2, class_name=cls_name, args=cfg)