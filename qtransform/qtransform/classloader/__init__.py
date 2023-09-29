from typing import Any
from types import ModuleType
from omegaconf import DictConfig
from logging import Logger
from qtransform.utils.introspection import get_classes

def get_data(log: Logger, package_name: ModuleType, class_name: str, parent_class: type) -> Any:
    """
        A generic implementation to dynamically load classes from a given package with a given name.
        It is used for dataset, tokenizer and model currently.
    """
    #get all classes which are subclasses of DatasetWrapper within own package context
    c = get_classes(package_name, parent_class)
    if class_name not in c:
        log.error(f"{parent_class.__name__} {class_name} not found in {package_name.__name__}")
        raise KeyError
    data_wrapper: Any = c[class_name]
    return data_wrapper