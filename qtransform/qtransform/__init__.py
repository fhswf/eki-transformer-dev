import logging
import os
from contextlib import contextmanager
from torch import device, cuda, backends
import torch
from abc import ABC
log = logging.getLogger(__name__)
from omegaconf import DictConfig


#idea stolen from:
#https://stackoverflow.com/questions/100003/what-are-metaclasses-in-python and https://refactoring.guru/design-patterns/singleton/python/example
class SingletonMeta(type, ABC):
    """
    Uses metaclasses in order to implement a singleton like structure.
    It keeps track of objects of a certain class and adds them to _instances
    if a class using SingletonMeta is instantiated.
    Each object of a class is instantiated only once.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

class ConfigSingleton(metaclass=SingletonMeta):
    """
    Config singleton in order to manipulate the config from different places without passing them as a reference each time.
    """
    _config: DictConfig

    def __init__(self):
        #no arguments to avoid having to call constructor with args each time config needs to be accessed
        #instead of: ConfigSingleton(config).config, we can use ConfigSingleton().config
        pass

    @property
    def config(self):
        return self._config
    
    @config.setter
    def config(self, value: DictConfig):
        if not isinstance(value, DictConfig):
            try:
                value = DictConfig(value)
            except:
                log.error(f'Config value invalid: "{value}"', exc_info=True)
        self._config = value

def get_module_config_path():
    return os.path.join('/'.join(__file__.split('/')[:-2]), 'qtransform' , 'conf')

def main(cfg):
    """Run this app like amodule, Note: cfg is a Hydra config (OmegaConf Object)"""
    from qtransform import  __main__ as _self
    _self.main(cfg)


def jls_extract_def(loglevel=logging.INFO):
    from hydra import initialize, initialize_config_module, initialize_config_dir, compose
    from omegaconf import OmegaConf
    import qtransform 
    import logging
    import yaml
    
    config_path = qtransform.get_module_config_path()
    with open(os.path.join(config_path, 'hydra','job_logging', 'custom.yaml'), 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    
    logging.config.dictConfig(config)
    logging.getLogger().setLevel(loglevel) 
    return initialize_config_dir, config_path, compose

def notebook_run(args, loglevel):
    initialize_config_dir, config_path, compose = jls_extract_def(loglevel)
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name="config.yaml", overrides=args)
        print(cfg)
        main(cfg)

@contextmanager
def with_hyrda(arg, loglevel):
    """creates a global cfg variable from hydra run config"""
    initialize_config_dir, config_path, compose = jls_extract_def(loglevel)
    with initialize_config_dir(version_base=None, config_dir=config_path):
        global cfg
        old_cfg = cfg
        cfg = compose(config_name="config.yaml", overrides=arg)
        log = logging.getLogger(__name__)
        log.info(f"Hydra compose config is: {cfg}")
        yield
        cfg = old_cfg

def with_config(arg, loglevel):
    """simiar to with_hyrda but a function a decorator for singular function definitions"""
    def wrapper_decorator(func):
        def wrapped_func(*args, **kwargs):
            initialize_config_dir, config_path, compose = jls_extract_def(loglevel)
            with initialize_config_dir(version_base=None, config_dir=config_path):
                cfg = compose(config_name="config.yaml", overrides=arg)
                log = logging.getLogger(__name__)
                log.info(f"Hydra compose config is: {cfg}")
                return func(cfg, *args, **kwargs)
        return wrapped_func
    return wrapper_decorator

        
class DeviceSingleton:
    """
        Boilerplate class which contains the device used for the entire process. The reason why a class is created is in order to monitor when changes
        to the device are going to occur by using the property decorator. Currently, changes will be allowed and logged.
    """
    def __init__(self):
        self._device = None
        pass

    @property
    def device(self):
        return self._device

    @device.setter
    def device(self, value):
        match value:
            case 'cuda':
                new_device = 'cuda' if cuda.is_available() else 'cpu'
            case 'gpu':
                new_device = 'cuda' if cuda.is_available() else 'cpu'
            case 'mps':
                new_device = 'mps' if backends.mps.is_available() else 'cpu'
            case 'cpu':
                new_device = 'cpu'
            case _:
                log.warning(f'Device {new_device} not recognized. Using default: CPU')
                new_device = 'cpu'
        self._device = device(new_device)
        log.info(f'Device specified: {value}. Using device: {new_device}')
        # torch.set_default_device(self._device) # does not work for dataloader forks....

device_singleton = DeviceSingleton()