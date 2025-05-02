from functools import wraps

import wandb
import os
from qtransform.utils.id import ID
import hydra
import omegaconf
import logging

_qtransform_wandb_conf = None
_WANDB_ENABLED = False


def wandb_enabled(func):
    """
    A decorator to check if wandb is enabled.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        global _WANDB_ENABLED
        if _WANDB_ENABLED:
            return func(*args, **kwargs)
        return None

    return wrapper

# some docs for hydra and wandb:
# https://wandb.ai/adrishd/hydra-example/reports/Configuring-W-B-Projects-with-Hydra--VmlldzoxNTA2MzQw

def wandb_ensure_envs(cfg):
    # if wandb is used, make sure we dont write caches and logs to invalid locations (such as on slurm); ~/.cache/wandb is default
    # WANDB_DIR is the current working dir, this sould be the same as hydras to avoid confusion 
    WANDB_CACHE_DIR = os.environ.get("WANDB_CACHE_DIR", None)
    if WANDB_CACHE_DIR is None:
        os.environ["WANDB_CACHE_DIR"] = os.path.join(os.getcwd(), ".wandb_cache")
        
    WANDB_CONFIG_DIR = os.environ.get("WANDB_CONFIG_DIR", None)
    if WANDB_CONFIG_DIR is None:
        os.environ["WANDB_CONFIG_DIR"] = os.path.join(os.getcwd(), ".wandb_config")
    
    # check working dirs, TODO maybe cwd should not change
    print(f"Working directory : {os.getcwd()}")
    print(f"Output directory  : {hydra.core.hydra_config.HydraConfig.get().runtime.output_dir}")
    
    WANDB_DIR = os.environ.get("WANDB_DIR", None)
    if WANDB_DIR is None:
        os.environ["WANDB_DIR"] = os.path.join(os.getcwd())

@wandb_enabled
def wandb_setup_logger(cfg):
    logger = logging.getLogger('wandb_logger')
    logger.setLevel(logging.root.level)
    for handler in logging.root.handlers:
        logger.addHandler(handler)
    pass

@wandb_enabled
def wandb_log(*args, **kwargs):
    wandb.log(*args, **kwargs)
    pass

def wandb_init(cfg, config=None):
    # prep for wandb usage
    if cfg.wandb.enabled:
        global _WANDB_ENABLED
        _WANDB_ENABLED = True
        wandb_ensure_envs(cfg)
        global _qtransform_wandb_conf
        _qtransform_wandb_conf = cfg 
        if "wandb_name" in cfg.keys():
            name = cfg["wandb_name"]
        else:
            name = ID
        if config is None:
            config = omegaconf.OmegaConf.to_container(cfg, resolve=True)

        wandb.init(
            name=name,
            entity=cfg.wandb.init.entity, 
            project=cfg.wandb.init.project,
            settings=wandb.Settings(start_method="thread", symlink=False),
            dir=hydra.core.hydra_config.HydraConfig.get().runtime.output_dir,
            config=config 
        )
    pass

@wandb_enabled
def wandb_finish():
    wandb.finish()
    pass

@wandb_enabled
def wandb_watch(*args, **kwargs):
    wandb.watch(*args, **kwargs)
    pass
