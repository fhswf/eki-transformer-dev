import datetime
import os
from typing import Any, Dict, Tuple, Union
import hydra
from omegaconf import DictConfig
import torch
import logging
from torch import nn
log = logging.getLogger(__name__)

def get_default_chkpt_folder() -> str:
    """
        Returns the default directory where model checkpoints are stored if the path was not configured
        by the user with the cfg variables "model_dir".
    """
    return os.path.join(os.getenv("HOME"), *__package__.split("."), "model_dir")

def load_checkpoint(cfg: DictConfig) -> Tuple[int, Union[Any, Dict]]:
    """ load torch model checkpoint"""
    #model_dir: specify custom path for checkpoints, otherwise use directory model_dir in qtransform/utils/model_dir
    chkpt_folder = get_default_chkpt_folder()
    if "from_checkpoint" not in cfg.run:
        log.error(f'Key "from_checkpoint" not specified in run config')
        raise KeyError()
    #from_checkpoint is the absolute path to a file, ignore model_dir 
    if os.path.isabs(cfg.run.from_checkpoint):
        chkpt_folder, from_checkpoint = os.path.split(cfg.run.from_checkpoint)
    elif "model_dir" in cfg.run:
        if os.path.isabs(cfg.run.model_dir):
            chkpt_folder = cfg.run.model_dir
            from_checkpoint = cfg.run.from_checkpoint
        else:
            #outputs are stored in qtransform/outputs/
            try:
                chkpt_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, "outputs", cfg.run.model_dir)
            except:
                chkpt_folder = os.getcwd()
            from_checkpoint = cfg.run.from_checkpoint
    checkpoint_path = os.path.join(chkpt_folder, from_checkpoint)
    if not os.path.isfile(checkpoint_path):
            log.error(f"Checkpoint {checkpoint_path} is not a file")
            raise FileNotFoundError
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

def save_checkpoint(cfg: DictConfig, 
    model: nn.Module, 
    optimizer, 
    timestamp:datetime, 
    metrics:Dict, 
    epoch:int, 
    model_cfg: Any,
    tokenizer_cfg: Any) -> str:
    """save torch model checkpoint from training, returns path to saved file."""
    
    chkpt_folder = get_default_chkpt_folder()
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
                "tokenizer_cfg": tokenizer_cfg, 
                "metrics": metrics,
                "quantized": True if cfg.get('quantization', {"quantize": False})["quantize"] is True else False
                }, f=checkpoint_path)
        log.info(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path