from types import ModuleType
import types 
from typing import Iterable
from omegaconf import DictConfig
from torch import nn
import logging
import pkgutil, inspect, importlib

log = logging.getLogger(__name__)
def get_classes(module: ModuleType, parent_class: ...):
    """
    'module' should be either a list of paths to look for
    modules in or a python module.
    """
    # type save guard, maybe relevant later...
    paths = None 
    package = None
    if isinstance(module, ModuleType):
        paths = module.__path__
        package = module.__name__
    else:
        log.error(f"module {module} is not an importable ModuleType")
        raise TypeError
    
    classes = {}
    for p in pkgutil.iter_modules(paths):
        m = importlib.import_module(package + "." + p[1])
        if hasattr(log,"trace"): log.trace(f"module found:  {m}")
        for name, obj in inspect.getmembers(m):
            if hasattr(log,"trace"): log.trace(f"class found:  {name}={obj}")
            if parent_class is not None: 
                if inspect.isclass(obj) and issubclass(obj, parent_class):
                    if name not in classes:
                        log.debug(f"nn.Module class found: {name}={obj}")
                        classes[name] = obj
            elif inspect.isclass(obj):
                if name not in classes:
                    log.debug(f"nn.Module class found: {name}={obj}")
                    classes[name] = obj
    return classes
    

