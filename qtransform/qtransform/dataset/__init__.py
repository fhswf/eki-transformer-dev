from typing import Any, Union, Dict
from omegaconf import DictConfig, open_dict
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes, concat_paths, get_dtype
import qtransform
from qtransform import classloader
from dataclasses import dataclass, fields
from enum import Enum
from os.path import join

log = logging.getLogger(__name__)

class DatasetRunType(Enum):
    TRAIN = "train"
    EVAL = "eval"
    BENCH = "bench"

@dataclass
class DatasetSizes:
    """
        Remember the size of datasets in percent to retain the size when shuffling and to 
        do checks before data should be loaded. 
    """
    train: float = 0.0 #size of training data
    eval: float = 0.0 #size of the subset of training data to check if model is overfitting
    test: float = 0.0
    bench: float = 0.0 

    def __post_init__(self):
        empty_split = 0 #check how many splits are not empty
        count = 0
        for field in fields(self):
            attr = getattr(self, field.name)
            if not isinstance(attr, float):
                log.error(f'{field.name} is not a floating point number.')
                raise KeyError()
            if attr < 0.0 or attr > 1.0:
                log.error(f'Data split {field.name} has to be within range 0.0 and 1.0, not {attr}')
                raise ValueError()
            if attr == 0.0:
                empty_split += 1
            count += 1
        if self.eval > 0.0 and self.train == 0.0:
            log.error(f'Cannot validate training if size of training data is empty')
            raise ValueError()
        elif self.eval == 1.0:
            log.warning(f'Size of validation split is equal to training split (eval = 1.0)')
        #all fields are 0.0, no point in continuing process as no data can be loaded
        if count == empty_split:
            log.error(f'Sizes of specified splits are zero.')
            raise ValueError()

@dataclass
class DatasetInfo:
    """
        Dataclass containing the datasets for training, eval, testing, benchmark along with the name of the dataset.
        After construction, a simple type check is done with the __post_init__ hook.
    """
    name: str
    train: Dataset = None
    eval: Dataset = None
    test: Dataset = None
    bench: Dataset = None

    """def __setattr__(self, __name, __value):
        if __name not in self.fields.keys():
            log.error(f'DatasetInfo should only contain fields: {self.fields.keys()}')
            raise KeyError()
        field = self.fields[__name]
        current_attr = getattr(self, field.name)
        if current_attr is not None and not isinstance(current_attr, field.type):
                log.error(f'DatasetInfo field {field.name} expects field type {field.type}, not {type(current_attr)}')
                raise TypeError()"""

    def __post_init__(self):
        self.fields = {x.name:x for x in fields(self)}
        for field in self.fields.values():
            current_attr = getattr(self, field.name)
            if current_attr is not None and not isinstance(current_attr, field.type):
                log.error(f'DatasetInfo field {field.name} expects field type {field.type}, not {type(current_attr)}')
                raise TypeError()

from abc import ABC, abstractmethod
class DatasetWrapper(ABC):
    """
    Capsule around Dataset, to unify their interfaces.
    Each DatasetWrapper has to contain a method to (down)load the data, create a Dataloader, 
    and provide information on whether the dataset contained in this wrapper provides training, eval/test or benchmark data.
    """

    dataset_sizes: DatasetSizes = None
    cfg: DictConfig = None

    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg
        if self.cfg.get('name') is None:
            log.error(f'No dataset name specified.')
            raise KeyError()
        if self.cfg.get('sizes') is None:
            log.warning(f'No sizes for the data splits specified.')
            raise KeyError()
        #add empty args property to avoid None checking every time
        if self.cfg.get('args') is None:
            with open_dict(self.cfg):
                self.cfg["args"] = {}
        self.dataset_sizes = DatasetSizes(**cfg.sizes)
        self.dataset_info = DatasetInfo(name=self.cfg.name)
        self.tokenized_dir = concat_paths([*cfg.dataset_dir, "tokenized", cfg.tokenizer.encoding])
        self.dataset_file = join(self.tokenized_dir, self.cfg.name+ '-' + self.cfg.tokenizer.dtype + '.bin')
        #currently, dtype has to be set by user. maybe it could also be automatically infered by the max tokens property of Tokenizer
        if self.cfg.tokenizer.get('dtype') is None:
            log.debug(f'Dtype for dataset omited. Assuming default: Int64')
            self.dtype = get_dtype('Int64')
        else:
            self.dtype = get_dtype(self.cfg.tokenizer.dtype)

    @classmethod
    @abstractmethod
    def load_dataset(self) -> DatasetInfo:
        """
            Loads a dataset from the config specified in the constructor. The split argument specifies
            the size of the returned dataset which have been stored in an instance of type DataSizes. 

            TODO:   Check if it leads to performance issues in terms of memory usage if Datasets for all
                    types (train, test, eval, bench) is created at once
        """
        pass

    def check_split(self, split: str):
        """
            Checks whether the DatasetSizes dataclass contains a field with name split.
        """
        splits = [x.name for x in fields(DatasetSizes)]
        if split not in splits:
            log.error(f'Datasets can only be split among {splits}, not {split}')
            raise ValueError()

    @classmethod
    @abstractmethod
    def shuffle(self):
        """
            Idea:   Have an instance of type dataset (named all_datasets), its size is the sum of all instantiated datasets in DatasetInfo
                    E.g. training, eval dataset each are 10MB, all_datasets is 20MB of size
                    When shuffle is called, the training and eval datasets are created from all_datasets, containing different tensors than before.
            TODO:   Test the behavior of non-sequential datasets (test dataset goes from 90-100% and from 0-10%)
            TODO:   check size of dataset and splits, pick random number with torch.rand
        """
        pass

def get_data(dataset_cfg: DictConfig) -> DatasetWrapper:
    import qtransform.dataset as package_self
    dataset_wrapper: DatasetWrapper = classloader.get_data(log, package_self, dataset_cfg.wrapper, DatasetWrapper, args={"cfg": dataset_cfg})
    return dataset_wrapper

def get_loader(dataloader_cfg: DictConfig, data:Dataset) -> DataLoader:
    log.debug(f"get_loader config: {dataloader_cfg}")
    loader = DataLoader(data, **dataloader_cfg)
    return loader