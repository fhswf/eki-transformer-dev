from logging import getLogger
import os
from contextlib import contextmanager
from torch import device, cuda, backends
log = getLogger(__name__)

def get_module_config_path():
    return os.path.join('/'.join(__file__.split('/')[:-2]), 'qtransform' , 'conf')

def main(cfg):
    """Run this app like amodule, Note: cfg is a Hydra config (OmegaConf Object)"""
    from qtransform import  __main__ as mn
    mn.main(cfg)


def jls_extract_def():
    from hydra import initialize, initialize_config_module, initialize_config_dir, compose
    from omegaconf import OmegaConf
    import qtransform 
    import logging
    import yaml
    
    config_path = qtransform.get_module_config_path()
    with open(os.path.join(config_path, 'hydra','job_logging', 'custom.yaml'), 'r') as stream:
        config = yaml.load(stream, Loader=yaml.FullLoader)
    
    logging.config.dictConfig(config)
    logging.getLogger().setLevel(logging.INFO) 
    return initialize_config_dir, config_path, compose

def notebook_run(args):
    initialize_config_dir, config_path, compose = jls_extract_def()
    with initialize_config_dir(version_base=None, config_dir=config_path):
        cfg = compose(config_name="config.yaml", overrides=args)
        print(cfg)
        main(cfg)

@contextmanager
def with_hyrda(arg):
    """creates a global cfg variable from hydra run config"""
    initialize_config_dir, config_path, compose = jls_extract_def()
    with initialize_config_dir(version_base=None, config_dir=config_path):
        global cfg
        old_cfg = cfg
        cfg = compose(config_name="config.yaml", overrides=arg)
        log = getLogger(__name__)
        log.info(f"Hydra compose config is: {cfg}")
        yield
        cfg = old_cfg

def with_config(arg):
    """simiar to with_hyrda but only for singular function definition"""
    def wrapper_decorator(func):
        def wrapped_func(*args, **kwargs):
            initialize_config_dir, config_path, compose = jls_extract_def()
            with initialize_config_dir(version_base=None, config_dir=config_path):
                cfg = compose(config_name="config.yaml", overrides=arg)
                log = getLogger(__name__)
                log.info(f"Hydra compose config is: {cfg}")
                func(cfg, *args, **kwargs)
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


device_singleton = DeviceSingleton()