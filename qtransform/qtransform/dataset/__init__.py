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
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
import qtransform.dataset as package_self
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
#TokenizedDatasetWrapper



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

class DataCollatorWrapper():
    pass

class TokenizedDatasetFactory():
    @classmethod
    def get_tokenized_data(cfg: DictConfig) -> (DatasetSplits, DataCollatorWrapper):
        tokenized_dataset_generator: TokenizedDatasetGenerator = get_tokenized_dataset_generator(cfg.tokenized.type, cfg)
        #tokenized_splits: [x for x in tokenized_dataset_generator.check_tokenized(cfg)
        #log.info(f'Tokenized splits: {[x for x in tokenized_splits if tokenized_splits[x] == True]}')
        tokenized_dataset_generator: TokenizedDatasetGenerator = get_tokenized_dataset_generator(cfg.untokenized.type, cfg)
        untokenized_data = tokenized_dataset_generator.get_untokenized_data()
        tokenized_data = tokenized_dataset_generator.tokenize_data(untokenized_data)

        #TODO: collator


    @classmethod
    def get_collator(cfg: DictConfig) -> DataCollatorWrapper:
        pass


from abc import ABC, abstractmethod
import os

class TokenizedDatasetGenerator(ABC):

    FILE_EXTENSION: str

    def __init__(self, cfg: DictConfig):
        super().__init__()
        log.info(f"TokenizedDatasetGenerator config:  {cfg}")
        self.cfg = cfg

    @classmethod
    @abstractmethod
    def get_tokenized_dataset(self,) -> DatasetSplits:
        raise NotImplementedError
    
    def get_cache_filepaths(self):
        cache_dir = concat_paths(cfg.cache_dir)
        cache_file_prefix = cfg.cache_filename_prefix
        splits = [x.name for x in fields(DatasetSplits)]
        return [os.path.join(cache_dir, cache_file_prefix + "-" + split + "-tokenized."+FILE_EXTENSION) for x in splits]

    @classmethod
    def check_tokenized(cfg: DictConfig) -> Dict[str,  bool]:
        
        #if splits do not exist in base untokenized dataset, entire untokenized dataset will be loaded 
        missing_splits = {x:os.path.exists() for x in splits}
        return missing_splits

    @classmethod
    @abstractmethod
    def tokenize_data(self, untokenized_data: DatasetDict) -> None:
        raise NotImplementedError
    
    @classmethod
    @abstractmethod
    def get_untokenized_data(self) -> DatasetDict:
        raise NotImplementedError

#TODO: factories and singletons could work here
class DataLoaderWrapper():

    def __init__(self, cfg):
        self.dataloader_cfg = cfg.dataloader
        log.info(f'Dataloader cfg: {self.dataloader_cfg}')
        self.tokenized_dataset_splits: DatasetSplits = TokenizedDatasetFactory.get_tokenized_data(cfg)
        self.collator = TokenizedDatasetFactory.get_collator(cfg)

    def get_loader(self, split: str) -> DataLoader:
        log.debug(f"get_loader config: {self.dataloader_cfg} for split {split}")
        # loader = DataLoader(data, generator=torch.Generator(device='cuda'), **dataloader_cfg) # does no work for dataloader forks
        kwargs = {**self.dataloader_cfg}
        if split=='train':
            kwargs['shuffle'] = False
        if self.cfg.get('collate_fn'):
            log.warning("TODO collate_fn via config is not supported yet")
        tokenizer = tokenizer_singleton.tokenizer
        #the way block_size is handled in TokenizedDatasetGenerator and DataCollator is iffy
        self.data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='max_length', max_length=self.cfg.tokenized.args.block_size)
        kwargs['collate_fn'] = self.data_collator
        #print(type(self.datasets[split]))
        #print(self.datasets[split])
        ds_split = self.tokenized_datasets.__dict__.get(split, None)
        if ds_split is None:
            log.warning(f"Split {split} not found in avaiable dataset splits. Usually train eval or bench.")
            return None
        else:
            loader = DataLoader(dataset=self.tokenized_datasets[split], **kwargs)
            log.debug(f'len of dataset loader: {len(loader)}')
            return loader



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

import numpy as np
from typing import Tuple
import torch

from qtransform.utils.introspection import load_class
def get_dataset_wrapper(dataset_cfg: DictConfig) -> DatasetWrapper:
    log.info(f"loading dataset wrapper {dataset_cfg.get('wrapper')} with config: {dataset_cfg}")
    return load_class(logger=log, module=qtransform.dataset, class_name=dataset_cfg.get("wrapper"), parent_class=DatasetWrapper, args={"cfg": dataset_cfg})

def get_tokenized_dataset_generator(generator_module: str, dataset_cfg: DictConfig) -> TokenizedDatasetGenerator:
    """
    Basically does the same thing as get_dataset_wrapper without having to specify the name of the wrapper class.
    """
    log.info(f"loading tokenized dataset generator from module {generator_module} with config: {dataset_cfg}")
    module_name = package_self.__package__ + "." + generator_module
    classes =  get_classes(package_self, parent_class=TokenizedDatasetGenerator).values()
    log.debug(f'Found classes: {classes}')
    found_class = list(filter(lambda x: x.__module__ == module_name , classes))
    if len(found_class) == 0:
        log.error(f'Could not find TokenizedDatasetGenerator in module: {module_name} with specified type: {generator_module}')
        raise ValueError()
    return found_class[0](dataset_cfg)

def get_loader(dataloader_cfg: DictConfig, data:Dataset) -> DataLoader:
    log.debug(f"get_loader config: {dataloader_cfg}")
    # loader = DataLoader(data, generator=torch.Generator(device='cuda'), **dataloader_cfg) # does no work for dataloader forks
    loader = DataLoader(data, **dataloader_cfg)
    log.debug(f'len: {len(loader)}')
    return loader
