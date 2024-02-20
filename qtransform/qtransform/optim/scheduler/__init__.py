from torch import optim
from torch.optim import lr_scheduler
from omegaconf import DictConfig, ListConfig
from logging import getLogger
from pprint import PrettyPrinter
from qtransform.classloader import get_data
from dataclasses import dataclass
from typing import Dict, Any, Union

log = getLogger(__name__)

#TODO: access hydra config within warmup as global variable
NUMBER_WARMUP_EPOCHS = None

def warmup(current_step: int):
    return 1 / (10 ** (float(NUMBER_WARMUP_EPOCHS - current_step)))

@dataclass
class SchedulerCfg:
    name: str
    args: Dict[str, Any]

def get_scheduler(optimizer: optim.Optimizer, scheduler_cfg: DictConfig) -> lr_scheduler.LRScheduler:
    """
    Creates a SequentialLR scheduler from a given optimizer along with optional schedulers. The schedulers along their params are specified
    in the hydra config. The order in which the schedulers are called is specified by the order of appearance in the hydra config.
    Example of scheduler config: 

      decay_lr : True
      warmup_iters : 100
      schedulers:
        - name: ConstantLR
          args: 
            factor: 0.1
            total_iters: 2
        - name: ExponentialLR
          args:
            gamma: 0.9
        - name: StepLR
          args:
            step_size: 1
    """
    log.debug(f'Getting scheduler')
    #warmup scheduler from mkohler's answer in https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch
    if not isinstance(optimizer, optim.Optimizer):
        log.error(f'Specified optimizer ({optimizer}) is not of type optim.Optimizer')
        raise KeyError()
    if not isinstance(scheduler_cfg, DictConfig):
        log.error(f'Specified scheduler config ({pprint.PrettyPrinter(indent=1).pformat(scheduler_cfg)}) is not of type omegaconf.DictConfig')
        raise KeyError()
    if not scheduler_cfg.decay_lr:
        return None
    #warmup_scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=warmup)
    schedulers = list()
    schedulers_cfg = scheduler_cfg.get('schedulers', list())
    if not isinstance(schedulers_cfg, Union[list, ListConfig]):
        log.error(f'Schedulers are not a list, but {type(scheduler_cfg)}')
        raise TypeError()
    #get torch scheduler from config
    for scheduler in schedulers_cfg:
        name = scheduler.get('name')
        args = scheduler.get('args')
        log.debug(f'Going through scheduler: {name} with args: {args}')
        #schedulers.append(get_data(log, lr_scheduler, name, lr_scheduler.LRScheduler, args=args))
        scheduler_class = getattr(lr_scheduler, name)
        schedulers.append(scheduler_class(**{"optimizer": optimizer, **args}))
    milestones = scheduler_cfg.get('milestones', list())
    if milestones is None:
        milestones = list()
    if not isinstance(milestones, Union[list, ListConfig]):
        log.error(f'Milestones are not a list, but of type: {type(scheduler_cfg)}.')
        raise TypeError()
    if len(schedulers) == 1:
        milestones = list() #ignore milestones
    #alternative: ConcatScheduler (https://pytorch.org/ignite/generated/ignite.handlers.param_scheduler.create_lr_scheduler_with_warmup.html)
    scheduler =  lr_scheduler.SequentialLR(optimizer, schedulers, milestones)
    return scheduler