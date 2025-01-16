import os
import hydra
import logging
from hydra.core.global_hydra import GlobalHydra
from dataclasses import fields
from types import ModuleType
from typing import  Any, Dict
import importlib
import pkgutil
import inspect
from inspect import isclass,isfunction
from modelflow.utils.id import ID 
log = logging.getLogger(__name__)

class _SingletonWrapper:
    """
    A singleton wrapper class. Its instances would be created for each decorated class. 
    """
    def __init__(self, cls):
        self.__wrapped__ = cls
        self._instance = None

    def __call__(self, *args, **kwargs):
        """Returns a single instance of decorated class"""
        if self._instance is None:
            self._instance = self.__wrapped__(*args, **kwargs)()
        return self._instance

    
def singleton(*args, **kwargs):
    """ A singleton decorator."""
    # check for correct usage with parentheses => @singleton()
    if len(args) > 0: 
        assert not isclass(args[0]) or not isfunction(args[0]) 
        
    def _singleton(cls):
        """ actual decorator function """
        assert isclass(cls) # wrap only classes
        return _SingletonWrapper(cls)
    
    return _singleton

def dataclass_from_dict(klass, d):
    try:
        fieldtypes = {f.name:f.type for f in fields(klass)}
        return klass(**{f:dataclass_from_dict(fieldtypes[f],d[f]) for f in d})
    except:  # noqa: E722
        return d # Not a dataclass field
    
#def load_module():
#    """Used load and instanciate class from omageconf launch config"""
#    pass

def get_classes(module: ModuleType, parent_class: type) -> Dict[str, Any]:
    """
    'module' should be either a list of paths to look for modules in or a python module.
    """
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

def load_class(module: ModuleType, parent_class: type, class_name:str,  args: Dict[str, Any] = None):
    """ 
    searches for classes in module with parent_class as parent.
    creates a class with class_name as its name and passes args to its __init__ constructor.
    """
    log.info(f"Loading class {module.__name__}.{class_name}(parent: {parent_class})")
    found_classes = get_classes(module, parent_class)
    log.debug(f'found_classes: {found_classes}')
    if class_name not in found_classes:
        log.error(f"{parent_class.__name__} {class_name} not found in {module.__name__}")
        raise KeyError
    cls: Any = found_classes[class_name]
    if args:
        log.info(f'Passing arguments {args} to class: {cls}')
        return cls(**args)    
    return cls()

# def ensure_folder():  if we need params here, create anopther wrapper around the wrapper
def ensure_folder(f):
    """ decorator to make sure that for functions that return a writable path, 
    the path actually exsists. """
    def wrapper(*args, **kwargs):
        path = f(*args, **kwargs)
        os.makedirs(path, exist_ok=True)
        return path
    return wrapper

# dont use cache here as pwd from hydra could change (even though it should not)
def get_cwd() -> str:
    cwd:str
    if GlobalHydra().is_initialized():
        cwd = str(hydra.core.hydra_config.HydraConfig.get().runtime.cwd)
    else:
        cwd = str(os.getcwd())
    return cwd

@ensure_folder
def get_output_dir() -> str:
    return os.path.join(get_cwd(), "modelflow")

@ensure_folder
def get_output_dir_with_id() -> str:
    return os.path.join(get_output_dir(), ID)

@ensure_folder
def get_output_debug_dir() -> str:
    return os.path.join(get_output_dir_with_id(), "debug")

@ensure_folder
def get_output_chkpt_dir() -> str:
    return os.path.join(get_output_dir_with_id(), "state")

@ensure_folder
def get_output_exports_dir() -> str:
    return os.path.join(get_output_dir_with_id(), "exports")

@ensure_folder
def get_output_analysis_dir() -> str:
    return os.path.join(get_output_dir_with_id(), "analysis")

@ensure_folder
def get_output_log_dir() -> str:
    return os.path.join(get_output_dir_with_id(), "logs")

@ensure_folder
def get_output_process_dir() -> str:
    return os.path.join(get_output_dir_with_id(), "process")