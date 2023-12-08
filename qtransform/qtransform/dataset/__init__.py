from typing import Any, Union, Dict
from omegaconf import DictConfig
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes
import qtransform.classloader
from dataclasses import dataclass, fields
from enum import Enum

log = logging.getLogger(__name__)

class DatasetRunType(Enum):
    TRAIN = "train"
    EVAL = "eval"
    BENCH = "bench"

@dataclass
class DatasetSizes:
    """
        Remember the size of datasets in order to keep the size when shuffling.
    """
    train: float = 0.8 #train on 80 percent of data
    eval: float = 0.1 #percentage of training dataset to be used for evaluation. default: 10 percent
    test: float = 0.2 #save 20 percent of dataset for inference
    bench: float = 1.0

    def __post_init__(self):
        for field in fields(self):
            attr = getattr(self, field.name)
            if not isinstance(attr, float):
                log.error(f'{field.name} is not a floating point number.')
                raise KeyError()
            if attr < 0.0:
                log.error(f'Cannot create {field.name} dataset of negative size')
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

    def __post_init__(self):
        self.fields = fields(self)
        for field in [x for x in self.fields]:
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
            log.error(f'No sizes for the datasets specified.')
            raise KeyError()
        self.dataset_sizes = DatasetSizes(**cfg.sizes)

    @classmethod
    @abstractmethod
    def load_dataset(self, split: str) -> DatasetInfo:
        """
            Loads a dataset from the config specified in the constructor. The split argument specifies
            the size of the returned dataset which have been stored in an instance of type DataSizes. 

            TODO:   Check if it leads to performance issues in terms of memory usage if Datasets for all
                    types (train, test, eval, bench) is created at once
        """
        pass

    @classmethod
    @abstractmethod
    def shuffle():
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
    dataset_wrapper: DatasetWrapper = qtransform.classloader.get_data(log, package_self, dataset_cfg.wrapper, DatasetWrapper, args=dataset_cfg)
    return dataset_wrapper

def get_loader(dataloader_cfg: DictConfig, data:Dataset) -> DataLoader:
    log.debug(f"get_loader config: {dataloader_cfg}")
    loader = DataLoader(data, **dataloader_cfg)
    return loader