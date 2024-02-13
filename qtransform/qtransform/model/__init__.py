from omegaconf import DictConfig
from qtransform.classloader import find_clas_and_create_instance
from torch import nn
import logging
log = logging.getLogger(__name__)

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    log.debug(f"get_model config: {model_cfg}")
    if model_cfg.get('cls') is None:
        log.error(f'No model class specified')
        raise KeyError()
    from qtransform import model as _model
    args = model_cfg.get("args")
    #models need to specify their hyperparameters in init parameter named "config"
    model = find_clas_and_create_instance(log, package_name = _model, class_name = model_cfg.get('cls'), parent_class = nn.Module)
    #construct model if no args have been given
    if args:
        model = model(config = args)
    else:
        model = model()
    return model