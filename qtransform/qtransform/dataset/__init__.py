from typing import Any
from omegaconf import DictConfig
import logging
import pkgutil, inspect, importlib
from torch.utils.data import Dataset, DataLoader
log = logging.getLogger(__name__)

def _get_module(module_name):
    """
    find and import the module "module_name" dynamicly in current directory/path
    """
    l = list(filter(lambda x: x.name==module_name, pkgutil.iter_modules(__path__)))
    log.debug(f"Found dataset module: {l}")
    if len(l) == 0:
        log.error(f"Dataset type module_name: {module_name} not found")
        log.error(f"Options are: {list(map(lambda x:x.name, pkgutil.iter_modules(__path__)))}")
        raise KeyError
    if len(l) > 1:
        log.critical(f"Found more then one module to import for {module_name}, is something wrong with the search path?")
        raise KeyError
    return importlib.import_module(__name__ + "." + l[0][1])

def get_data(dataset_cfg: DictConfig) -> Any:
    """ load data from module and config spec. """
    log.debug(f"get_data config: {dataset_cfg}")
    m = _get_module(dataset_cfg.module)
    if not hasattr(m, "load_dataset"):
        log.critical(f"module {m} does not have a 'load_dataset' function")
        raise NotImplementedError
    if "args" in dataset_cfg:
        return m.load_dataset(dataset_cfg.name, dataset_cfg)
    else:
        return m.load_dataset(dataset_cfg.name, dataset_cfg)

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
    def load_dataset():
        pass
    def get_dataloader():
        pass