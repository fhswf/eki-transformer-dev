
from omegaconf import DictConfig
from torch import nn
import logging
import pkgutil, sys, inspect, importlib

log = logging.getLogger(__name__)
def get_model_classes():
    """
    using python introspection, 
    this function find subclasses of troch.nn.Module conntained in this module.
    """
    defined_models = {}
    for p in pkgutil.iter_modules(__path__):
        module = importlib.import_module(__name__ + "." + p[1])
        if hasattr(log,"trace"): log.trace(f"module found:  {module}")
        for name, obj in inspect.getmembers(module):
            if hasattr(log,"trace"): log.trace(f"class found:  {name}={obj}")
            if inspect.isclass(obj) and issubclass(obj, nn.Module):
                log.debug(f"nn.Module class found: {name}={obj}")
                defined_models[name] = obj
    return defined_models

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    models = get_model_classes()
    model = models[model_cfg.cls](model_cfg.args)
    return model