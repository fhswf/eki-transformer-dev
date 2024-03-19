from qtransform.tokenizer import Tokenizer, get_tokenizer
from typing import Union, Dict
from omegaconf import DictConfig
from logging import getLogger

log = getLogger(__name__)

class TokenizerSingleton():
    """
    Singleton used to access tokenizer from different places during execution time.
    """
    _tokenizer: Tokenizer = None
    _cfg: DictConfig = None #unsure if cfg is necessary when tokenizer instance is created

    def __init__(self):
        self._tokenizer = None
        self._cfg = None

    @property
    def tokenizer(self):
        return self._tokenizer

    @tokenizer.setter
    def tokenizer(self, value: Tokenizer):
        if isinstance(value, Union[Dict, DictConfig]):
            value = get_tokenizer(value)
        elif not isinstance(value, Union[Tokenizer, None]):
            raise TypeError(f'Tokenizer is not of type Tokenizer or None')
        self._tokenizer = value

#no idea how to make class with properties into a singleton without constructing objects
tokenizer_singleton = TokenizerSingleton()