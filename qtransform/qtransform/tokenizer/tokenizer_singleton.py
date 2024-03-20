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
    def tokenizer(self, value: Union[Dict, DictConfig, Tokenizer]):
        if isinstance(value, Union[Dict, DictConfig]):
            value: Tokenizer = get_tokenizer(value)
        elif not isinstance(value, Union[Tokenizer, None]):
            raise TypeError(f'Tokenizer is not of type Tokenizer or None')
        self._tokenizer: Tokenizer = value

    @property
    def cfg(self):
        return self._cfg

    @cfg.setter
    def cfg(self, value: Union[Dict, DictConfig]):
        if not isinstance(value, Union[Dict, DictConfig]):
            raise TypeError(f'cfg is not of type Dict or DictConfig')
        self.tokenizer = value
        self._tokenizer: Tokenizer = value

#no idea how to make class with properties into a singleton without constructing objects
#idea: metaclasses (https://refactoring.guru/design-patterns/singleton/python/example)
tokenizer_singleton = TokenizerSingleton()