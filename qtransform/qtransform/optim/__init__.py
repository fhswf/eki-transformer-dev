

## example for combining multiple sheduler

##https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch 
#train_scheduler = CosineAnnealingLR(optimizer, num_epochs)
#
#def warmup(current_step: int):
#    return 1 / (10 ** (float(number_warmup_epochs - current_step)))
#warmup_scheduler = LambdaLR(optimizer, lr_lambda=warmup)
#
#scheduler = SequentialLR(optimizer, [warmup_scheduler, train_scheduler], [number_warmup_epochs])
from logging import getLogger
from torch import optim
from torch.nn import Module
from qtransform.classloader import get_data
from inspect import signature
from omegaconf import DictConfig, open_dict
from .scheduler import get_scheduler
log = getLogger(__name__)


#(log, package_name, class_name, parent_class)
def get_optim(model: Module, optim_cfg: DictConfig) -> optim.Optimizer:
    """
        Dynamically creates an instance of torch.optim.Optimizer with the
        arguments supplied in the hydra config. Only arguments which are supported by
        the optimizer (namely from its constructor) will be applied, others will be logged as a warning.
    """
    if not isinstance(model, Module):
        log.error(f'Specified model ({model}) is not of type torch.nn.Module')
        raise KeyError()
    if not isinstance(optim_cfg, DictConfig):
        log.error(f'Specified config ({optim_cfg}) is not of type omegaconf.DictConfig')
        raise KeyError()
    if not isinstance(optim_cfg.optimizer, str):
        log.error(f'Invalid optimizer specified ({optim_cfg.optimizer}).')
        raise KeyError()
    elif optim_cfg.args is None:
        log.warning(f'No arguments for optimizer {optim_cfg.optimizer} defined.')
    #change name of learning_rate to lr to pass it to the optimizer constructor
    if optim_cfg.args.get("lr") is None and optim_cfg.args.learning_rate is not None:
        with open_dict(optim_cfg):
            optim_cfg.args.update({"lr": optim_cfg.args.pop("learning_rate")})
    optimizer_cls = get_data(log, optim, optim_cfg.optimizer, optim.Optimizer)
    #find out which args from config can be applied to optimizer
    init_args_cls = signature(optimizer_cls.__init__)
    applicable_cfg_args = set(optim_cfg.args.keys()) & set(init_args_cls.parameters.keys())
    log.debug(f'Configurable optimizer args: {applicable_cfg_args}')
    non_applicable_cfg_args = set(optim_cfg.args.keys()) - set(["decay_lr"]) - set(init_args_cls.parameters.keys())
    if len(non_applicable_cfg_args) > 0:
        log.warning(f'Arguments {non_applicable_cfg_args} are not supported by optimizer {optim_cfg.optimizer}. Skipping those arguments for now.')
    #create instance of optimizer with args from hydra config
    return optimizer_cls(params=model.parameters(), **{x:optim_cfg.args.get(x) for x in applicable_cfg_args})