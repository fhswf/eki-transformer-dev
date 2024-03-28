from dataclasses import dataclass
#from qtransform.tokenizer #TODO: dataclass for tokenizer config
from omegaconf import DictConfig
from typing import Union, Dict

#TODO: config of splits depends on generator.
@dataclass
class UntokenizedSplits():
    pass

@dataclass
class TokenizationArgs():
        cache_dir: str#directorry where cached datasets are stored. default: ~/.cache/huggingface
        data_column_name: str = "text" #name of the column that contains the training data. usually "text"
        batches: int = 1000 #split dataset into shards to perform tokenization more efficiently
        chunking: bool = True #if True, split long sentences after chunk_size characters for faster tokenization. Default: False
        chunk_size: int = 100 

@dataclass
class UntokenizedData():
    type: str
    splits: UntokenizedSplits
    args: TokenizationArgs