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
    m = _get_module(dataset_cfg.module)
    if not hasattr(m, "load_dataset"):
        log.critical(f"module {m} does not have a 'load_dataset' function")
        raise NotImplementedError
    if "args" in dataset_cfg:
        return m.load_dataset(dataset_cfg.name, dataset_cfg.args)
    else:
        return m.load_dataset(dataset_cfg.name)

def get_loader(dataloader_cfg: DictConfig, data:Dataset) -> DataLoader:
    loader = DataLoader(data, **dataloader_cfg)
    return loader

