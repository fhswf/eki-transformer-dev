import datetime
import os
from typing import Any, Dict, Tuple, Union
import hydra
from omegaconf import DictConfig
import torch
import logging
from torch import nn
from stat import S_ISFIFO
from hydra.core.global_hydra import GlobalHydra
from qonnx.core.modelwrapper import ModelWrapper
# maybe only do this when it is required, for this howiever is always the case
from onnx.shape_inference import infer_shapes
from dataclasses import dataclass, field, asdict, fields
from pprint import PrettyPrinter
from qtransform import ConfigSingleton
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


def get_default_chkpt_folder() -> str:
    """
        Returns the default directory where model checkpoints are stored if the path was not configured
        by the user with the cfg variables "checkpoint_dir".
    """
    return os.path.join(os.getenv("HOME"), *__package__.split("."), "checkpoint_dir")


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
    return os.path.join(get_cwd(), "output", "qtransform")

@ensure_folder
def get_output_debug_dir() -> str:
    return os.path.join(get_output_dir(), "debug")

@ensure_folder
def get_output_chkpt_dir() -> str:
    return os.path.join(get_output_dir(), "chkpts")

@ensure_folder
def get_output_exports_dir() -> str:
    return os.path.join(get_output_dir(), "exports")

@ensure_folder
def get_output_analysis_dir() -> str:
    return os.path.join(get_output_dir(), "analysis")

@ensure_folder
def get_output_log_dir() -> str:
    return os.path.join(get_output_dir(), "logs")


def load_onnx_model(path: str) -> ModelWrapper:
    """
    Loads ONNX model from a filepath. 
    """
    if not isinstance(path, str):
        log.error(f'Could not load ONNX model because: path {path} is not a string.')
    if not os.path.isfile(path):
        log.error(f'Could not load ONNX model because: path {path} is not a file.')
    log.info(f'Loading ONNX model from "{path}"')
    #qonnx lib also works with onnx models
    model = ModelWrapper(path)    
    try:
        model = infer_shapes(model.model)
        return ModelWrapper(model)
    except Exception as e:
        log.warning(f"shape infernce faild due to {e}. Continue without infer_shapes. Good luck!")
    return model

def write_to_pipe(pipe_name: str, content: str) -> None:
    """
    Write into a named pipe in order for other qtransform processes to access information from this current instance.
    The filepath of checkpoints or ONNX models could be written into the pipe in order to continue training from it or 
    perform inference or benchmarking. 

    The "pipe" field from the hydra config specifies the filepath of the named pipe (by default: /dev/null). If the pipe does
    not exist yet and the current operating system is UNIX-like, it will be created. 
    """
    #by default, write checkpoint to /dev/null if pipe name is omited
    if not isinstance(pipe_name, str):
        log.debug(f'Invalid type for pipe: {pipe_name}. Using /dev/null')
        pipe_name = '/dev/null'
    if not os.path.exists(pipe_name):
        from sys import platform
        if platform == "win32":
            log.error(f'Cannot create pipes on non-UNIX system.')
            raise RuntimeError()
        log.debug(f'Creating named pipe "{pipe_name}"')
        os.mkfifo(pipe_name)

    if pipe_name == '/dev/null': #avoid logging when output goes nowhere
        return
    elif not S_ISFIFO(os.stat(pipe_name).st_mode):
        log.error(f'Specified filepath "{pipe_name}" is not a pipe.')
    else:
        log.info(f'Writing content "{content}" into fifo "{pipe_name}". ' \
                 f'Until another process reads from the fifo, the current process (PID {os.getpid()}) is blocked.'
        )
        #writing into a named pipe blocks process until another process reads from it
        #problematic if something should be done after writing into the pipe
        with open(pipe_name, 'w') as pipe:
            pipe.write(content)
