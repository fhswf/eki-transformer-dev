from torch import optim
from torch.optim import lr_scheduler
from omegaconf import DictConfig
from logging import getLogger

log = getLogger(__name__)

def warmup(current_step: int):
    return 1 / (10 ** (float(number_warmup_epochs - current_step)))

def get_scheduler(optimizer: optim.Optimizer, scheduler_cfg: DictConfig) -> lr_scheduler.LRScheduler:
    #warmup scheduler from mkohler's answer in https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
    if not isinstance(optimizer, optim.Optimizer):
        log.error(f'Specified optimizer ({optimizer}) is not of type optim.Optimizer')
        raise KeyError()
    if not isinstance(scheduler_cfg, DictConfig):
        log.error(f'Specified scheduler ({scheduler_cfg}) is not of type omegaconf.DictConfig')
        raise KeyError()
    if not scheduler_cfg.decay_lr:
        return None
    warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    