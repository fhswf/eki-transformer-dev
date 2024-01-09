import datetime
import os
from typing import Any, Dict, Tuple, Union
import hydra
from omegaconf import DictConfig
import torch
import logging
from torch import nn
log = logging.getLogger(__name__)

def load_checkpoint(cfg: DictConfig) -> Tuple[int, Union[Any, Dict]]:
    """ load torch model checkpoint"""
    chkpt_folder = os.path.join(os.getenv("HOME"), *__package__.split("."), "model_dir")
    if not cfg.run.from_checkpoint:
        log.error(f"cfg.run.from_checkpoint is None, if you intent to load a checkpoint please specify the path here")
        raise FileNotFoundError
    if "model_dir" in cfg.run:
        if os.path.isabs(cfg.run.from_checkpoint):
            checkpoint_path = cfg.run.from_checkpoint
            if not os.path.isfile(checkpoint_path):
                log.error(f"Checkpoint {checkpoint_path} is not a file")
        elif os.path.isabs(cfg.run.model_dir) and not os.path.isabs(cfg.run.from_checkpoint):
            chkpt_folder = cfg.run.model_dir
            checkpoint_path = os.path.join(chkpt_folder, cfg.run.from_checkpoint) 
            if not os.path.isfile(checkpoint_path):
                log.error(f"Checkpoint {checkpoint_path} is not a file")
                raise FileNotFoundError
        elif not os.path.isabs(cfg.run.model_dir) and not os.path.isabs(cfg.run.from_checkpoint):
            try:
                chkpt_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, "outputs", cfg.run.model_dir)
            except:
                chkpt_folder = os.getcwd()
            checkpoint_path = os.path.join(chkpt_folder, cfg.run.from_checkpoint)

    else: # model_dir not present => need full abs path to checkpoint 
        if not os.path.isfile(cfg.run.from_checkpoint):
            log.error(f"Checkpoint {checkpoint_path} is not a file")
            raise FileNotFoundError
        checkpoint_path = cfg.run.from_checkpoint

    log.info(f"Loading checkpoint from {checkpoint_path}")
    from_epoch = 0    
    checkpoint = torch.load(checkpoint_path)
    if 'epoch' in checkpoint:
        from_epoch = checkpoint['epoch']
    else:
        try:
            i = str(cfg.run.from_checkpoint).index("epoch:")
            import re
            p = re.compile("[0-9]+")
            from_epoch = int(p.search(str(cfg.run.from_checkpoint)[i:]).group(0))
        except ValueError or AttributeError:
            log.warn("Modelcheckpint does not contain epoch information")
    return from_epoch,checkpoint

def save_checkpoint(cfg: DictConfig, model: nn.Module, optimizer, timestamp:datetime, metrics:Dict, epoch:int, model_cfg: Any) -> str:
    """save torch model checkpoint from training, returns path to saved file."""
    chkpt_folder = os.path.join(os.getenv("HOME"), *__package__.split("."), "model_dir")
    if "model_dir" in cfg.run:
        if os.path.isabs(cfg.run.model_dir):
            chkpt_folder = cfg.run.model_dir
        else:
            try:
                chkpt_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, "outputs", cfg.run.model_dir)
            except:
                chkpt_folder = os.getcwd()
    os.makedirs(chkpt_folder, exist_ok=True)
    if epoch % cfg.run.save_epoch_interval == 0:
        checkpoint_path = os.path.join(chkpt_folder,f'{cfg.model.cls}_{timestamp}__epoch:{epoch}')
        torch.save(obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "model_cfg": model_cfg,
                "metrics": metrics,
                }, f=checkpoint_path)
        log.info(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path