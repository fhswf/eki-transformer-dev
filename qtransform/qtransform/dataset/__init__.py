from typing import Any, Union, Dict
from omegaconf import DictConfig, open_dict
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes, concat_paths, get_dtype
import qtransform
from qtransform import classloader
from dataclasses import dataclass, fields
from enum import Enum
from os import listdir
from os.path import join
from qtransform.tokenizer import Tokenizer, get_tokenizer
import datasets
from datasets.dataset_dict import DatasetDict
from qtransform.dataset.untokenized import UntokenizedData

log = logging.getLogger(__name__)

class DatasetRunType(Enum):
    TRAIN = "train"
    EVAL = "eval"
    BENCH = "bench"


@dataclass
class DatasetSplits:
    """
        Dataclass containing the datasets for training, eval, testing, benchmark along with the name of the dataset.
        After construction, a simple type check is done with the __post_init__ hook.
    """
    train: Dataset = None
    eval: Dataset = None
    bench: Dataset = None

    # make class subscritable aka: self['train'] works
    def __getitem__(self, item):
        return getattr(self, item)
    
    def __setitem__(self, index, item):
        setattr(self, index, item)

    """def __setattr__(self, __name, __value):
        if __name not in self.fields.keys():
            log.error(f DatasetSplits should only contain fields: {self.fields.keys()}')
            raise KeyError()
        field = self.fields[__name]
        current_attr = getattr(self, field.name)
        if current_attr is not None and not isinstance(current_attr, field.type):
                log.error(f DatasetSplits field {field.name} expects field type {field.type}, not {type(current_attr)}')
                raise TypeError()"""

    def __post_init__(self):
        self.fields = {x.name:x for x in fields(self)}
        for field in self.fields.values():
            current_attr = getattr(self, field.name)
            if current_attr is not None and not isinstance(current_attr, field.type):
                log.error(f'DatasetSplits field {field.name} expects field type {field.type}, not {type(current_attr)}')
                raise TypeError()

from abc import ABC, abstractmethod
import os

class TokenizedDatasetGenerator(ABC):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__()
        log.info(f"TokenizedDatasetGenerator config:  {cfg}")
        self.cfg = cfg

    @abstractmethod
    def get_tokenized_dataset(self, *args, **kwargs) -> DatasetSplits:
        raise NotImplementedError
    
    @abstractmethod
    def prepare_data(self) -> None:
        raise NotImplementedError

class DataLoaderWrapper():
    def __init__(self):
        pass
    def check_tokenized(self):
        pass
    def get_loader(self):
        pass


class DatasetWrapper(ABC):
    def __init__(self, cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__()
        log.info(f"DatasetWrapper config:  {cfg}")
        self.cfg = cfg
        self.datasets: DatasetSplits = DatasetSplits()

    @abstractmethod
    def load_dataset(self, *args, **kwargs) -> Any:
        raise NotImplementedError
    
    @abstractmethod
    def get_loader(self, split: str) -> DataLoader:
        raise NotImplementedError

#TODO: OldDatasetWrapper tokenizes by memmap. find a good way to abstract this
#      another alternative would be to store huggingface apache arrow datasets
class OldDatasetWrapper(DatasetWrapper):
    """
    Capsule around Dataset, to unify their interfaces.
    Each DatasetWrapper has to contain a method to (down)load the data, create a Dataloader, 
    and provide information on whether the dataset contained in this wrapper provides training, eval/test or benchmark data.
    """

    cfg: DictConfig = None

    def __init__(self, cfg: DictConfig):
        #TODO: decide on dataset cache path
        self.dataset_info = DatasetSplits()
        self.tokenized_dir = concat_paths([*cfg.dataset_dir, "tokenized", cfg.tokenizer.encoding])
        

import numpy as np
from typing import Tuple
import torch

from qtransform.utils.introspection import load_class

def get_data(dataset_cfg: DictConfig) -> OldDatasetWrapper:
    return load_class(logger=log, module=qtransform.dataset, class_name=dataset_cfg.wrapper, parent_class=OldDatasetWrapper, args={"cfg": dataset_cfg})

def get_dataset_wrapper(dataset_cfg: DictConfig) -> DatasetWrapper:
    log.info(f"loading dataset wrapper {dataset_cfg.get('wrapper')} with config: {dataset_cfg}")
    return load_class(logger=log, module=qtransform.dataset, class_name=dataset_cfg.get("wrapper"), parent_class=DatasetWrapper, args={"cfg": dataset_cfg})

def get_loader(dataloader_cfg: DictConfig, data:Dataset) -> DataLoader:
    log.debug(f"get_loader config: {dataloader_cfg}")
    # loader = DataLoader(data, generator=torch.Generator(device='cuda'), **dataloader_cfg) # does no work for dataloader forks
    loader = DataLoader(data, **dataloader_cfg)
    log.debug(f'len: {len(loader)}')
    return loader
