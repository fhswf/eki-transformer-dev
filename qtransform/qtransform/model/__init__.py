from omegaconf import DictConfig
from qtransform.classloader import get_data
from torch import nn
import logging
log = logging.getLogger(__name__)

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    log.debug(f"get_model config: {model_cfg}")
    from qtransform import model as _model
    args = model_cfg.get("args")
    model = get_data(log, package_name = _model, class_name = model_cfg.get('cls'), parent_class = nn.Module, args=args)
    #construct model if no args have been given
    if not args:
        model = model()
    return model