from omegaconf import DictConfig
from qtransform.utils.introspection import get_classes
from torch import nn
import logging
log = logging.getLogger(__name__)

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    from qtransform import model as _model
    models = get_classes(_model,nn.Module)
    #models = get_model_classes()
    if "args" in model_cfg:
        model = models[model_cfg.cls](model_cfg.args)
    else: 
        model = models[model_cfg.cls]()
    return model