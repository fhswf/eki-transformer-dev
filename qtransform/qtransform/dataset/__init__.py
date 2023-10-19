from typing import Any
from omegaconf import DictConfig
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes
import qtransform.classloader
log = logging.getLogger(__name__)

def get_data(dataset_cfg: DictConfig) -> Dataset:
    import qtransform.dataset as package_self
    dataset_wrapper: DatasetWrapper = qtransform.classloader.get_data(log, package_self, dataset_cfg.wrapper, DatasetWrapper)
    return dataset_wrapper.load_dataset(dataset_cfg)

def get_loader(dataloader_cfg: DictConfig, data:Dataset) -> DataLoader:
    log.debug(f"get_loader config: {dataloader_cfg}")
    loader = DataLoader(data, **dataloader_cfg)
    return loader


from dataclasses import dataclass
from enum import Enum
class DatasetRunType(Enum):
    TRAIN = "train"
    EVAL = "eval"
    BENCH = "bench"

@dataclass
class DatasetInfo:
    name: str
    train: bool
    eval: bool
    test: bool
    bench: bool
    pass

from abc import ABC, abstractclassmethod
class DatasetWrapper(ABC):
    """
    Capsule around Dataset, to unify their interfaces.
    Each DatasetWrapper has to contain a method to (down)load the data, create a Dataloader, 
    and provide information on whether the dataset contained in this wrapper provides training, eval/test or benchmark data.
    """
    @abstractclassmethod
    def load_dataset(cfg: DictConfig) -> Dataset:
        pass
    def get_dataloader():
        pass