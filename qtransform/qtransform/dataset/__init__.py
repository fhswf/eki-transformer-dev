from typing import Any
from omegaconf import DictConfig
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes
log = logging.getLogger(__name__)


def deprecated_get_data(dataset_cfg: DictConfig) -> Any:
    """ load data from module and config spec. it does so by importing a module within its own package and calling 
        the load_dataset method directly.
    """
    log.debug(f"get_data config: {dataset_cfg}")
    #TODO: split module into module: (custom|torchvision,huggingface...) and module_name 
    m = _get_module(dataset_cfg.module, __name__, __path__)
    log.debug(f'loaded module: {m}')
    #c = get_classes(m, Dataset)
    if not hasattr(m, "load_dataset"):
        log.critical(f"module {m} does not have a 'load_dataset' function")
        raise NotImplementedError
    if "args" in dataset_cfg:
        return m.load_dataset(dataset_cfg.name, dataset_cfg)
    else:
        return m.load_dataset(dataset_cfg.name, dataset_cfg)

def get_data(dataset_cfg: DictConfig) -> Dataset:
    log.debug(f"get_data config: {dataset_cfg}")
    import qtransform.dataset as package_self
    #get all classes which are subclasses of DatasetWrapper within own package context
    c = get_classes(package_self, DatasetWrapper)
    if dataset_cfg.module not in c:
        log.error(f"DatasetWrapper {dataset_cfg.module} not found in {package_self.__package__}")
        raise KeyError
    dataset_wrapper: DatasetWrapper = c[dataset_cfg.module]
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