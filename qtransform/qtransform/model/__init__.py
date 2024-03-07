from omegaconf import DictConfig
from qtransform.classloader import get_data
from torch import nn
import logging
log = logging.getLogger(__name__)
from dataclasses import dataclass

@dataclass
class ModelSupport():
    "Static information about capabilities of a model class"
    training: bool= False

class ModelInfoMixin():
    support = ModelSupport()
    def __init__(self):
        pass

    @classmethod
    def supports(cls) -> ModelSupport():
        return cls.support

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    log.debug(f"get_model config: {model_cfg}")
    if model_cfg.get('cls') is None:
        log.error(f'No model class specified')
        raise KeyError()
    from qtransform import model as _model
    args = model_cfg.get("args")
    #models need to specify their hyperparameters in init parameter named "config"
    model = get_data(log, package_name = _model, class_name = model_cfg.get('cls'), parent_class = nn.Module)
    #construct model if no args have been given
    if args:
        model = model(config = args)
    else:
        model = model()
    return model


def get_onnx(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    log.debug(f"get_model config: {model_cfg}")
    if model_cfg.get('cls') is None:
        log.error(f'No model class specified')
        raise KeyError()
    from qtransform import model as _model
    args = model_cfg.get("args")
    #models need to specify their hyperparameters in init parameter named "config"
    model = get_data(log, package_name = _model, class_name = model_cfg.get('cls'), parent_class = nn.Module)
    #construct model if no args have been given
    if args:
        model = model(config = args)
    else:
        model = model()
    return model