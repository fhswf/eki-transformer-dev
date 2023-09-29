from omegaconf import DictConfig
from qtransform.classloader import get_data
from torch import nn
import logging
log = logging.getLogger(__name__)

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    log.debug(f"get_model config: {model_cfg}")
    from qtransform import model as _model
    model = get_data(log, _model, model_cfg.cls, nn.Module)
    if "args" in model_cfg:
        model = model(model_cfg.args)
    else: 
        model = model()
    return model