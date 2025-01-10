import os
import hydra
import logging
from hydra.core.global_hydra import GlobalHydra
from dataclasses import fields

from inspect import isclass,isfunction
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
    
# def ensure_folder():  if we need params here, create anopther wrapp

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

# @ensure_folder
# def get_output_dir() -> str:
#     return os.path.join(get_cwd(), "outputs")
# 
# @ensure_folder
# def get_output_debug_dir() -> str:
#     return os.path.join(get_output_dir(), "debug")
# 
# @ensure_folder
# def get_output_chkpt_dir() -> str:
#     return os.path.join(get_output_dir(), "chkpts")
# 
# @ensure_folder
# def get_output_exports_dir() -> str:
#     return os.path.join(get_output_dir(), "exports")
# 
# @ensure_folder
# def get_output_analysis_dir() -> str:
#     return os.path.join(get_output_dir(), "analysis")
# 
# @ensure_folder
# def get_output_log_dir() -> str:
#     return os.path.join(get_output_dir(), "logs")
