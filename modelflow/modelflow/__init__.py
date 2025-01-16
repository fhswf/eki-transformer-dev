import logging
import os
from contextlib import contextmanager
from modelflow.utils.helper import singleton
from omegaconf import DictConfig
log = logging.getLogger(__name__)

def get_module_config_path():
    return os.path.join('/'.join(__file__.split('/')[:-2]), 'modelflow' , 'conf')

def main(cfg):
    """Run this app like amodule, Note: cfg is a Hydra config (OmegaConf Object)"""
    from modelflow import  __main__ as _self
    _self.main(cfg)


def jls_extract_def(loglevel=logging.INFO):
    from hydra import initialize_config_dir, compose
    import modelflow 
    import logging
    import yaml
    
    config_path = modelflow.get_module_config_path()
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


    
@singleton()
class CFG(object):
    """Stores Config from Hydra compose"""
    def __init__(self, cfg:DictConfig) -> None:
        self.cfg = cfg
        pass
    
    def get_cfg(self):
        return self.cfg
    
    def __call__(self):
        return self.get_cfg()
    
    def __repr__(self) -> str:
        return self.get_cfg()
    
    def __str__(self) -> str:
        return self.get_cfg()
