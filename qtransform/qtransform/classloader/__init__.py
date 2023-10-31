from typing import Any
from types import ModuleType
from logging import Logger
from qtransform.utils.introspection import get_classes


def get_data(log: Logger, package_name: ModuleType, class_name: str, parent_class: type, args = None) -> Any:
    """
        A generic implementation to dynamically load classes from a given package with a given name.
        It is used for dataset, tokenizer and model currently.
        The returned class is either a wrapper or a model and can be configured with the args parameter.
    """
    #get all classes which are subclasses of DatasetWrapper within own package context
    log.debug(f"Loading class {package_name.__name__}.{class_name}(parent: {parent_class})")
    c = get_classes(package_name, parent_class)
    if class_name not in c:
        #log.error(f"{parent_class.__name__} {class_name} not found in {package_name.__name__}")
        raise KeyError
    data_wrapper: Any = c[class_name]
    #construct an object with given parameters
    #TODO: consider if this is a good idea
    if args:
        return data_wrapper(args)    
    return data_wrapper