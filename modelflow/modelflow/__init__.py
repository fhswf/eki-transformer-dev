import logging
import os
from contextlib import contextmanager
from modelflow.utils.helper import singleton
from omegaconf import DictConfig
from typing import KeysView, Any, ItemsView, Iterator
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
    
    def get_dict_cfg(self):
        return self.cfg
    
    def __call__(self):
        return self.get_dict_cfg()
    
    def __repr__(self) -> str:
        return str(self.get_dict_cfg())
    
    def __str__(self) -> str:
        return str(self.get_dict_cfg())

    def __setitem__(self, key: Any, value: Any) -> None:
        return self.cfg.__setitem__(key, value)

    def __setattr__(self, key: str, value: Any) -> None:
        if 'cfg' in self.__dict__.keys():
            return self.__dict__['cfg'].__setattr__(key, value)
        else:
            return self.__dict__.update({key: value})

    def __getattr__(self, key: str) -> Any:
        if 'cfg' in self.__dict__.keys():
            return self.__dict__['cfg'].__getattr__(key)
        else:
            return self.__dict__[key]

    def __getitem__(self, key: Any) -> Any:
        return self.cfg.__getitem__(key)

    def __delattr__(self, key: str) -> None:
        return self.cfg.__delattr__(key)

    def __delitem__(self, key: Any) -> None:
        return self.cfg.__delitem__(key)

    def get(self, key: Any, default_value: Any = None) -> Any:
        return self.cfg.get(key, default_value=default_value)

    def keys(self) -> KeysView[Any]:
        return self.cfg.keys()

    def __contains__(self, key: object) -> bool:
        return self.cfg.__contains__(key)

    def __iter__(self) -> Iterator[Any]:
        return self.cfg.__iter__()

    def items(self) -> ItemsView[Any, Any]:
        return self.cfg.items()

    def __eq__(self, other: Any) -> bool:
        return self.cfg.__eq__(other=other)

    def __ne__(self, other: Any) -> bool:
        return self.cfg.__ne__(other=other)