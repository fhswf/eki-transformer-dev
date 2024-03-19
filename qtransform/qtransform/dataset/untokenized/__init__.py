from dataclasses import dataclass
#from qtransform.tokenizer #TODO: dataclass for tokenizer config
from omegaconf import DictConfig
from typing import Union, Dict

@dataclass
class SplitNames():
    train: str = "train"
    eval: str = "validation"
    bench: str = "test"

    def __setattr__(self, name, value):
        if not isinstance(value, str):
            raise TypeError(f'Cannot set {name} to non-str value {value}')
        self.__dict__[name] = value

@dataclass
class SplitSizes():
    train: float = 0.0 #size of training data
    eval: float = 0.0 #size of the subset of training data to check if model is overfitting
    bench: float = 0.0 

    def __setattr__(self, name, value):
        if not isinstance(value, float):
            raise TypeError(f'Cannot set {name} to non-float value {value}')
        self.__dict__[name] = value

@dataclass
class Splits():
    _names: SplitNames
    _sizes: SplitSizes

    @property
    def names(self):
        return self._names
    @names.setter
    def names(self, value):
        if isinstance(value, Union[Dict, DictConfig]):
            value = SplitNames(**value)
        elif not isinstance(value, SplitNames):
            raise TypeError(f'names can only be of type: Union[Dict, DictConfig, SplitNames]')
        self._names = value

    @property
    def sizes(self):
        return self._sizes
    @sizes.setter
    def sizes(self, value):
        if isinstance(value, Union[Dict, DictConfig]):
            value = SplitSizes(**value)
        elif not isinstance(value, SplitSizes):
            raise TypeError(f'names can only be of type: Union[Dict, DictConfig, SplitSizes]')
        self._sizes = value

@dataclass
class TokenizationArgs():
        cache_dir: str#directorry where cached datasets are stored. default: ~/.cache/huggingface
        data_column_name: str = "text" #name of the column that contains the training data. usually "text"
        batches: int = 1000 #split dataset into shards to perform tokenization more efficiently
        chunking: bool = True #if True, split long sentences after chunk_size characters for faster tokenization. Default: False
        chunk_size: int = 100 

@dataclass
class UntokenizedDatasetName():
    path: str
    args: DictConfig = None 
    def __post_init__(self):
        if self.args is None:
            self.args = DictConfig({})

@dataclass
class UntokenizedData():
    type: str 
    _name: UntokenizedDatasetName
    _splits: Splits
    _tokenization_args: TokenizationArgs
    tokenizer_cfg: DictConfig

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value: Union[Dict, DictConfig, UntokenizedDatasetName]):
        if isinstance(value, Union[Dict, DictConfig]):
            value = UntokenizedDatasetName(**value)
        elif not isinstance(value, UntokenizedDatasetName):
            raise TypeError(f'Name can only be of type: Union[Dict, DictConfig, UntokenizedDatasetName]')
        self._name = value
    
    @property
    def splits(self):
        return self._splits

    @splits.setter
    def splits(self, value: Union[Dict, DictConfig, Splits]):
        if isinstance(value, Union[Dict, DictConfig]):
            value = Splits(**value)
        elif not isinstance(value, Splits):
            raise TypeError(f'Splits can only be of type: Union[Dict, DictConfig, Splits]')
        self._splits = value

    @property
    def tokenization_args(self):
        return self._tokenization_args

    @tokenization_args.setter
    def tokenization_args(self, value: Union[Dict, DictConfig, TokenizationArgs]):
        if isinstance(value, Union[Dict, DictConfig]):
            value = TokenizationArgs(**value)
        elif not isinstance(value, TokenizationArgs):
            raise TypeError(f'tokenization_args can only be of type: Union[Dict, DictConfig, TokenizationArgs]')
        self._tokenization_args = value