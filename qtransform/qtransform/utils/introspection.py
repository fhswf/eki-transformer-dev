from types import ModuleType
from typing import Iterable
from omegaconf import DictConfig
from torch import nn
import logging
import pkgutil, inspect, importlib
import sys
from os.path import join

log = logging.getLogger(__name__)
def get_classes(module: ModuleType, parent_class: type):
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
                        log.debug(f"{parent_class.__name__} class found: {name}={obj}")
                        classes[name] = obj
            elif inspect.isclass(obj):
                if name not in classes:
                    log.debug(f"{parent_class.__name__} class found: {name}={obj}")
                    classes[name] = obj
    return classes
    
def _get_module(module_name, package_name = None, scope = None):
    """
    find and import the module "module_name" dynamically either in current directory/path if scope variable is set to the path
    or from all known python modules (sys.path) if scope = None
    in order to correctly import a module with importlib.import_module, the name of the package in python notation must be provided
    (package.).
    package_name  is the name of the package in python notation (__name__), 
    scope is the path to the package (__path__)
    """
    if scope == None:
        scope = sys.path
    log.debug(f'searching for module {module_name} within scope: {scope}')
    #__path__ makes the search limit to only the current module, not all modules under sys.path
    l = list(filter(lambda x: x.name==module_name, pkgutil.iter_modules(scope)))
    log.debug(f"Found dataset module: {l}")
    if len(l) == 0:
        log.error(f"Dataset type module_name: {module_name} not found")
        log.error(f"Options are: {list(map(lambda x:x.name, pkgutil.iter_modules(scope)))}")
        raise KeyError
    if len(l) > 1:
        log.critical(f"Found more than one module to import for {module_name}, is something wrong with the search path?")
        raise KeyError
    _module_name = l[0].name
    if package_name == None:
        #import the module found from the first package in sys.path
        return importlib.import_module(_module_name)
    return importlib.import_module('..' + _module_name, package_name+'.')

def concat_paths(paths: list) -> str:
    """
        Concatinates paths of a list by joining them with os.path.join
    """
    if len(paths) == 0:
        raise ValueError("Error, cannot concatinate paths when argument is empty.")
    main_path = ""
    for path in paths:
        main_path = join(main_path, path)
    return main_path