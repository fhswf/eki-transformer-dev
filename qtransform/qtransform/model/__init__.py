from omegaconf import DictConfig, open_dict
from qtransform.classloader import get_data
from torch import nn
import logging
log = logging.getLogger(__name__)
from transformers import GPT2LMHeadModel 

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    log.debug(f"get_model config: {model_cfg}")
    if model_cfg.get('cls') is None:
        log.error(f'No model class specified')
        raise KeyError()
    pretrained = model_cfg.get('pretrained', False)
    if pretrained: 
        assert model_cfg.version is not None, f'pretrained model version needs to be specified'
        model = GPT2LMHeadModel.from_pretrained(model_cfg.version)
        #necessary for dataloader to retrieve exactly one full context input
        with open_dict(model_cfg):
            model_cfg.args.block_size = model.config.n_positions
            model_cfg.args.vocab_size = model.config.vocab_size
            model_cfg.args.calc_loss_in_model = False
        return model
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