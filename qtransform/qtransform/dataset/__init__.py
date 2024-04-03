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

#TODO: type of each split is still not clearly defined. For huggingface, it is a huggingface dataset. For files, it is a torch Dataset.
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
        if len(status_untokenized) > 0:
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

#TODO: no check if intermediate results of tokenization exist (tokenized but not grouped for huggingface)
class TokenizedDatasetGenerator(ABC):

    #contains file extension and other distinguishing factors (e.g. block_size, tokenized or grouped..)
    #split is prepended to suffix -> (cache_file_prefix, split, DATASET_FILE_SUFFIX)
    DATASET_FILE_SUFFIX: str
    DUMP_FILE_PATH: str #path for intermediate result of tokenization (e.g. tokenized but not grouped). not every generator needs to use this
    DATASET_FILE_PATH: str #by default: cache_dir from config
    CACHE_FILENAME_PREFIXES: Dict[DatasetSplitType, str]

    def __init__(self, cfg: DictConfig):
        """
        Defines the approach to retrieve tokenized data and tokenize raw data. Each implementation of TokenizedDatasetGenerator
        defines this for one specific type such as huggingface, raw files or other internet sources.

        In order to unify the parameters and return types of the different approaches, huggingface's
        DatasetDict is expected to be used in some way or another.
        """ 
        super().__init__()
        log.info(f"TokenizedDatasetGenerator config:  {cfg}")
        self.cfg = cfg
        if self.cfg.name_args is None:
            with open_dict(self.cfg):
                self.cfg.name_args = {}
        self.block_size = cfg.tokenized.args.block_size
        #field "name_args" from default config not included
        self.DATASET_FILE_PATH = concat_paths(self.cfg.tokenized.cache_dir)
        makedirs(self.DATASET_FILE_PATH, exist_ok=True)
        cache_filename_prefix = self.cfg.tokenized.cache_filename_prefix
        #TODO: create function which composes filename from list and adds a seperator (-) to avoid if else statements
        if self.cfg.tokenized.cache_filename_prefix[-1] != "-":
            cache_filename_prefix += "-"
        self.CACHE_FILENAME_PREFIXES = {split: cache_filename_prefix + split.name + "-" for split in DatasetSplitType}
        log.debug(f'CACHE_FILENAME_PREFIXES: {self.CACHE_FILENAME_PREFIXES}')

    @abstractmethod
    def get_tokenized_dataset(self) -> DatasetSplits:
        """
        Return all tokenized datasets and wrap them inside of a DatasetSplits instance.
        If splits are not found, an error is thrown instead of tokenizing immediately. 
        Use the method tokenize_dataset() for that and check if tokenized datasets are found with 
        the method check_tokenized().
        """
        raise NotImplementedError

    #TODO: create filename from list, seperating each entrry with "-"
    def check_tokenized(self) -> Dict[DatasetSplitType,  Dict[str, Union[bool, str]]]:
        """
        Checks for tokenized files under the path composed in the hydra config.
        The filepath is defined in the get_filepath_split function.
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
        filepath_splits =  {split: self.get_filepath_split(split) for split in splits}
        status_splits = {split:{"exists": os.path.exists(filepath_splits[split]), "filepath": filepath_splits[split]} for split in splits}
        return status_splits


    @abstractmethod
    def get_intermediate_tokenized_data(self) -> Dict[DatasetSplitType,  Dict[str, Union[bool, str]]]:
        """
        Checks if untokenized data has been processed in some way. If it does exist, that data should be tokenized.
        If it does not exist, the raw data from get_untokenized_data is fetched and processed.
        """
        raise NotImplementedError

    @abstractmethod
    def tokenize_data(self, untokenized_data: DatasetDict) -> None:
        """
        Tokenizes raw data alongside the approach implemented within the specific TokenizedDatasetGenerator class.
        It is expected that the split names of "untokenized_data" should have the names of DatasetSplitType (TRAIN, EVAL, BENCH).
        """
        #slice spits: https://huggingface.co/docs/datasets/loading#slice-splits
        #keys of DatasetDict are split types
        raise NotImplementedError
    
    @abstractmethod
    def get_untokenized_data(self, splits: List[DatasetSplitType]) -> datasets.DatasetDict:
        raise NotImplementedError

    @abstractmethod
    def get_collator(self) -> Callable:
        raise NotImplementedError

    def get_filepath_split(self,split: DatasetSplitType) -> str:
        """
        Gets the filepath of a split by joining the attributes
        DATASET_FILE_PATH, CACHE_FILENAME_PREFIXES and DATASET_FILE_SUFFIX.
        They should usually be set within the hydra config.

        Arguments:
        split: The split of type DatasetSplitType. The different splits are differentiated by the split name at the end of the filename prefix.

        """
        return os.path.join(self.DATASET_FILE_PATH, self.CACHE_FILENAME_PREFIXES[split] + self.DATASET_FILE_SUFFIX)

class DataLoaderWrapper():

    def __init__(self, dataset_cfg: DictConfig):
        self.dataloader_cfg = dataset_cfg.dataloader
        data_and_collator = TokenizedDatasetFactory.get_tokenized_data(dataset_cfg)
        self.tokenized_dataset_splits: DatasetSplits = data_and_collator[0]
        log.debug(f'Dataset: {self.tokenized_dataset_splits}')
        self.collate_fn: Callable = data_and_collator[1]
        device = device_singleton.device
        if device.type == 'cuda':
            cuda_kwargs = {'pin_memory': True,}
            #struct flag of dictconf prevents additional keys to be added (https://omegaconf.readthedocs.io/en/latest/usage.html#struct-flag)
            with open_dict(self.dataloader_cfg):
                self.dataloader_cfg.update(cuda_kwargs)
        log.info(f'Dataloader cfg: {self.dataloader_cfg}')

    def get_loader(self, split: DatasetSplitType) -> DataLoader:
        loader = DataLoader(dataset=self.tokenized_dataset_splits[split], **{"collate_fn": self.collate_fn, **self.dataloader_cfg})
        log.debug(f'len of dataset loader: {len(loader)}')
        return loader
    
    def get_data_format(self):
        """
        Get information about how the input ids and labels are stored within each sample.
        For example, huggingface uses a dictionary while files uses a tuple.
        """


def get_tokenized_dataset_generator(generator_module: str, dataset_cfg: DictConfig) -> TokenizedDatasetGenerator:
    """
    Basically does the same thing as get_dataset_wrapper without having to specify the name of the wrapper class.
    """
    log.info(f'loading tokenized dataset generator from module "{generator_module}"')
    module_name = package_self.__package__ + "." + generator_module
    #bug with get_classes that retrieves the last TokenizedDatasetGenerator class with the same name
    #some kind of check that names are unique could be useful
    classes =  get_classes(package_self, parent_class=TokenizedDatasetGenerator).values()
    log.debug(f'Found classes: {classes}')
    found_class = list(filter(lambda x: x.__module__ == module_name , classes))
    if len(found_class) == 0:
        log.error(f'Could not find TokenizedDatasetGenerator in module: {module_name} with specified type: {generator_module}')
        raise ValueError()
    return found_class[0](dataset_cfg)
