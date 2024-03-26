from typing import Any, Union, Dict, Callable, Tuple, List
from omegaconf import DictConfig, open_dict
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes, concat_paths, get_dtype
import qtransform
from qtransform import classloader
from dataclasses import dataclass, fields, InitVar
from enum import IntEnum
from os import listdir, makedirs
from os.path import join, exists
from qtransform.tokenizer import Tokenizer, get_tokenizer
import datasets
from datasets.dataset_dict import DatasetDict
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
import qtransform.dataset as package_self
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from qtransform import device_singleton
#TokenizedDatasetWrapper
import numpy as np
from typing import Tuple
import torch
from pprint import PrettyPrinter

#TODO: PrettyPrinter(indent=1).pformat(content) everywhere

log = logging.getLogger(__name__)


#TODO: python 3.11 could support StrEnum to avoid having to use .name.lower() for split name
class DatasetSplitType(IntEnum):
    TRAIN = 0
    EVAL = 1
    BENCH = 2


@dataclass
class DatasetSplits:
    """
        Dataclass containing the datasets for training, eval, testing, benchmark along with the name of the dataset.
    """
    splits: InitVar[Dict[DatasetSplitType, Dataset]] = None
    
    def __init__(self):
        #access splits with Enum values
        #DatasetSplits.eval not possible, DatasetSplits[DatasetSplitType.EVAL] possible
        self.splits = {
            DatasetSplitType.TRAIN: None,
            DatasetSplitType.EVAL: None,
            DatasetSplitType.BENCH: None
        }
    
    # make class subscritable aka: self['train'] works
    def __getitem__(self, item):
        return self.splits[item]

    #TODO: setting split to any value theoretically possible
    def __setitem__(self, index, item):
        self.splits[index] = item


class TokenizedDatasetFactory():

    def get_tokenized_data(cfg: DictConfig) -> (DatasetSplits, Callable):
        """
        Loads tokenized data and returns a DatasetSplits instance along a collate_fn for torch Dataloaders.
        If the tokenized data does not exist locally, the untokenized data is processed.
        The configuration for untokenized and tokenized data differs primarily in the format of data,
        e.g. tokenize a huggingface dataset into a numpy array.


        Arguments:
        cfg: Config containing dataset fields (e.g. tokenized and untokenized)

        Returns:
            Tuple of DatasetSplits and Callable (collate_fn)
        """
        dataset_splits = DatasetSplits()
        #generators
        tokenized_data_fetcher: TokenizedDatasetGenerator = get_tokenized_dataset_generator(cfg.tokenized.type, cfg)
        log.debug(f'Tokenized data fetcher: ${tokenized_data_fetcher.__class__.__name__}')
        untokenized_data_fetcher: TokenizedDatasetGenerator = get_tokenized_dataset_generator(cfg.untokenized.type, cfg)
        log.debug(f'Untokenized data fetcher: ${untokenized_data_fetcher.__class__.__name__}')

        #check if datasets exist
        status_splits = tokenized_data_fetcher.check_tokenized()
        log.debug(f'status_splits: {PrettyPrinter(indent=1).pformat(status_splits)}')
        status_untokenized = [split for split, status in status_splits.items() if status["exists"] is False]
        log.info(f'Splits "{[x.name for x in status_untokenized]}" do not exist. Tokenizing now.')
        
        #tokenize datasets
        untokenized_splits = untokenized_data_fetcher.get_untokenized_data(splits=status_splits)
        if len(untokenized_splits) > 0:
            tokenized_data_fetcher.tokenize_data(untokenized_splits)
        tokenized_splits = tokenized_data_fetcher.get_tokenized_dataset()
        collator_fn = tokenized_data_fetcher.get_collator()

        return (tokenized_splits, collator_fn)



from abc import ABC, abstractmethod
import os

class TokenizedDatasetGenerator(ABC):

    #contains file extension and other distinguishing factors (e.g. block_size, tokenized or grouped..)
    #split is prepended to suffix -> (cache_file_prefix, split, DATASET_FILE_SUFFIX)
    DATASET_FILE_SUFFIX: str
    DUMP_FILE_PATH: str #path for intermediate result of tokenization (e.g. tokenized but not grouped). not every generator needs to use this
    DATASET_FILE_PATH: str #by default: cache_dir from config
    CACHE_FILENAME_PREFIXES: Dict[DatasetSplitType, str]

    def __init__(self, cfg: DictConfig):
        super().__init__()
        log.info(f"TokenizedDatasetGenerator config:  {cfg}")
        self.cfg = cfg
        if self.cfg.name_args is None:
            with open_dict(self.cfg):
                self.cfg.name_args = {}
        #field name_args by default not included
        self.DATASET_FILE_PATH = concat_paths(self.cfg.cache_dir)
        makedirs(self.DATASET_FILE_PATH, exist_ok=True)
        cache_filename_prefix = self.cfg.cache_filename_prefix
        #TODO: create function which composes filename from list and adds a seperator (-) to avoid if else statements
        if self.cfg.cache_filename_prefix[-1] != "-":
            cache_filename_prefix += "-"
        self.CACHE_FILENAME_PREFIXES = {split: cache_filename_prefix + split.name + "-" for split in DatasetSplitType}
        log.debug(f'CACHE_FILENAME_PREFIXES: {self.CACHE_FILENAME_PREFIXES}')

    @abstractmethod
    def get_tokenized_dataset(self) -> DatasetSplits:
        raise NotImplementedError

    #TODO: create filename from list, seperating each entrry with "-"
    def check_tokenized(self) -> Dict[DatasetSplitType,  Dict[str, Union[bool, str]]]:
        """
        Checks for tokenized files under the path composed in the hydra config.
        cache_dir, <optional name_args depending on the datasetgenerator>, cache_file_prefix, split, dataset_suffix
        the suffix is defined in each dataset generator.
        os.path.join(self.DATASET_FILE_PATH, self.cache_file_prefix + "-" + split.name.lower() + "-" +self.DATASET_FILE_SUFFIX
        {
            <DatasetSplitType.TRAIN: 0>: {
                'exists': False or True,
                'filepath': 'cfg-path-to-file'
            },
            <DatasetSplitType.EVAL: 1>: {
                'exists': False or True,
                'filepath': 'cfg-path-to-file'
            },
            <DatasetSplitType.BENCH: 2>: {
                'exists': False or True,
                'filepath': 'cfg-path-to-file'
            },
        }
        """
        splits = [split for split in DatasetSplitType]
        filepath_splits =  {split: os.path.join(self.DATASET_FILE_PATH, self.CACHE_FILENAME_PREFIXES[split] +self.DATASET_FILE_SUFFIX) 
            for split in splits}
        status_splits = {split:{"exists": os.path.exists(filepath_splits[split]), "filepath": filepath_splits[split]} for split in splits}
        return status_splits

    @abstractmethod
    def tokenize_data(self, untokenized_data: DatasetDict) -> None:
        #slice spits: https://huggingface.co/docs/datasets/loading#slice-splits
        #keys of DatasetDict are split types
        raise NotImplementedError
    
    @abstractmethod
    def get_untokenized_data(self, splits: List[DatasetSplitType]) -> datasets.DatasetDict:
        raise NotImplementedError

    @abstractmethod
    def get_collator(self) -> Callable:
        pass

    def get_filepath_split(split: DatasetSplitType) -> str:
        return self.DATASET_FILE_PATH + self.CACHE_FILENAME_PREFIXES[split] + self.DATASET_FILE_SUFFIX

#TODO: factories and singletons could work here
class DataLoaderWrapper():

    def __init__(self, cfg):
        self.dataloader_cfg = cfg.dataloader
        data_and_collator = TokenizedDatasetFactory.get_tokenized_data(cfg)
        self.tokenized_dataset_splits: DatasetSplits = data_and_collator[0]
        self.collate_fn: Callable = data_and_collator[1]
        device = device_singleton.device
        if device.type == 'cuda':
            cuda_kwargs = {'pin_memory': True,}
            #struct flag of dictconf prevents additional keys to be added (https://omegaconf.readthedocs.io/en/latest/usage.html#struct-flag)
            with open_dict(cfg.dataset.dataloader):
                cfg.dataset.dataloader.update(cuda_kwargs)
        log.info(f'Dataloader cfg: {self.dataloader_cfg}')

    def get_loader(self, split: DatasetSplitType) -> DataLoader:
        loader = DataLoader(dataset=self.tokenized_datasets[split], **{"collate_fn": self.collate_fn, **self.dataloader_cfg})
        log.debug(f'len of dataset loader: {len(loader)}')
        return loader


def get_tokenized_dataset_generator(generator_module: str, dataset_cfg: DictConfig) -> TokenizedDatasetGenerator:
    """
    Basically does the same thing as get_dataset_wrapper without having to specify the name of the wrapper class.
    """
    log.info(f'loading tokenized dataset generator from module "{generator_module}"')
    module_name = package_self.__package__ + "." + generator_module
    classes =  get_classes(package_self, parent_class=TokenizedDatasetGenerator).values()
    log.debug(f'Found classes: {classes}')
    found_class = list(filter(lambda x: x.__module__ == module_name , classes))
    if len(found_class) == 0:
        log.error(f'Could not find TokenizedDatasetGenerator in module: {module_name} with specified type: {generator_module}')
        raise ValueError()
    return found_class[0](dataset_cfg)


############################delete later

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


def get_loader(dataloader_cfg: DictConfig, data:Dataset) -> DataLoader:
    log.debug(f"get_loader config: {dataloader_cfg}")
    # loader = DataLoader(data, generator=torch.Generator(device='cuda'), **dataloader_cfg) # does no work for dataloader forks
    loader = DataLoader(data, **dataloader_cfg)
    log.debug(f'len: {len(loader)}')
    return loader



from qtransform.utils.introspection import load_class
def get_dataset_wrapper(dataset_cfg: DictConfig) -> DatasetWrapper:
    log.info(f"loading dataset wrapper {dataset_cfg.get('wrapper')} with config: {dataset_cfg}")
    return load_class(logger=log, module=qtransform.dataset, class_name=dataset_cfg.get("wrapper"), parent_class=DatasetWrapper, args={"cfg": dataset_cfg})
