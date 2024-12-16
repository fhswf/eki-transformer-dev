from types import ModuleType
import logging
import pkgutil, inspect, importlib
import sys
from os.path import join
from numpy import dtype as np_dtype
from os.path import expanduser
from typing import Dict, Any
log = logging.getLogger(__name__)

def load_class(logger: logging.Logger, module: ModuleType, parent_class: type, class_name:str,  args: Dict[str, Any] = None):
    """ 
    searches for classes in module with parent_class as parent.
    creates a class with class_name as its name and passes args to its __init__ constructor.
    """
    logger.info(f"Loading class {module.__name__}.{class_name}(parent: {parent_class})")
    found_classes = get_classes(module, parent_class)
    log.debug(f'found_classes: {found_classes}')
    if class_name not in found_classes:
        logger.error(f"{parent_class.__name__} {class_name} not found in {module.__name__}")
        raise KeyError
    cls: Any = found_classes[class_name]
    if args:
        logger.info(f'Passing arguments {args} to class: {cls}')
        return cls(**args)    
    return cls()

def get_classes(module: ModuleType, parent_class: type) -> Dict[str, Any]:
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
        m = importlib.import_module(package + "." + p[1]) #p[1] is the module name
        for name, obj in inspect.getmembers(m):
            if parent_class is not None: 
                if inspect.isclass(obj) and issubclass(obj, parent_class):
                    if name not in classes:
                        if hasattr(log, "trace"): log.trace(f"{parent_class.__name__} class found: {name}={obj}")
                        classes[name] = obj
            elif inspect.isclass(obj):
                if name not in classes:
                    if hasattr(log, "trace"): log.trace(f"{parent_class.__name__} class found: {name}={obj}")
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
        Concatinates paths of a list by joining them with os.path.join.
        In order to avoid issues, the shortcut ~ is translated into the home directory of the user 
        executing the module.
    """
    if paths is None or len(paths) == 0:
        raise ValueError("Error, cannot concatinate paths when argument is empty.")
    main_path = ""
    paths[0] = paths[0].replace('~', expanduser('~'))
    for path in paths:
        main_path = join(main_path, path)
    return main_path

def get_dtype(dtype_alias: str) -> np_dtype:
    dtype = None
    try:
        dtype = np_dtype(dtype_alias)
    except:
        log.critical(f'Datatype {dtype_alias} not found within numpy datatype scope')
        raise KeyError()
    return dtype

from typing import Union, get_origin, get_args, List, Dict
def get_optional_type(_type):
    """
        Unwraps a type which might be encapsulated in type "Optional" from the typing package.
    """
    field_origin = get_origin(_type)
    if field_origin is Union:
        field_type = get_args(_type)
        #get first datatype of union, assuming that is is Optional, the second type is None anyway
        origin_type = field_type[0] if len(field_type) == 2 else field_type#and isinstance(field.type, Union) else field_type
    elif field_origin is None:
        #type was never wrapped
        origin_type = _type
    else: 
        origin_type =  field_origin
    return origin_type

def concat_strings(strings: List[str]) -> str:
    """
        Concats a list of immutable strings by joining them together. 
    """
    return ''.join(strings)
