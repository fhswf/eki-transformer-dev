import datetime
import os
from typing import Any, Dict, Tuple, Union
from omegaconf import DictConfig
import torch
import logging
from torch import nn
from pprint import PrettyPrinter
from qtransform import ConfigSingleton
from qtransform.utils import ID
from qtransform.utils.helper import get_output_chkpt_dir
log = logging.getLogger(__name__)

def load_checkpoint(from_file: Union[Dict, DictConfig]) -> Tuple[int, Union[Any, Dict]]:
    """ load torch model checkpoint and return the epoch count"""
    checkpoint_path = from_file.get_filepath()
    if not os.path.exists(checkpoint_path):
        log.error(f'Checkpoint {checkpoint_path} does not exist')
        raise FileNotFoundError()
    log.info(f"Loading checkpoint from {checkpoint_path}")
    from_epoch = 0    
    from qtransform import device_singleton
    checkpoint = torch.load(checkpoint_path, map_location=device_singleton.device)
    if 'epoch' in checkpoint:
        from_epoch = checkpoint['epoch']
    else:
        raise NotImplementedError("epoch needs to be in checkpoint for now")
        # try:
        #     i = str(filepath).index("epoch:")
        #     import re
        #     p = re.compile("[0-9]+")
        #     from_epoch = int(p.search(str(cfg.run.from_checkpoint)[i:]).group(0))
        # except ValueError or AttributeError:
        #     log.warn("Modelcheckpint does not contain epoch information")
    return from_epoch,checkpoint

def load_state_dict_proxy(model, checkpoint, **kwargs):
    """same as torch load state dict, however this check env for extra params. """
    strict = bool(int(os.environ.get("qtransform_load_state_strict", 1)))
    if "strict" not in kwargs.keys():
        kwargs.update({"strict": strict})
    return model.load_state_dict(checkpoint, **kwargs)

def save_checkpoint(model: nn.Module, 
    optimizer,
    timestamp:datetime, 
    metrics:Dict, 
    epoch:int,
    steps: int,
    **kwargs) -> str:
    """save torch model checkpoint from training, returns path to saved file."""
    
    cfg = ConfigSingleton().config
    dataset_name = cfg.dataset.name
    model_cfg=cfg.model
    log.debug(f'model_cfg when saving checkpoint: {PrettyPrinter(indent=1).pformat(model_cfg)}')
    tokenizer_cfg=cfg.tokenizer
    quant_cfg = cfg.get('quantization', None)
    # get name for encoding, if present in config
    if cfg.model.get("model_name", None) is not None:
        filename = f"{cfg.model.get('model_name')}_{dataset_name.replace('/', '__')}_{timestamp}__epoch:{epoch}"
    else:
        filename = f"{cfg.runtime.choices.model}_{dataset_name.replace('/', '__')}_{timestamp}__epoch:{epoch}"

    checkpoint_path = os.path.join(get_output_chkpt_dir(), filename)
    log.info(f"Model checkpoint saving to {checkpoint_path}")
    torch.save(obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_cfg": model_cfg,
            "tokenizer_cfg": tokenizer_cfg, 
            "metrics": metrics,
            "steps": steps,
            "quant_cfg": quant_cfg, 
            "quantized": True if quant_cfg.get("quantize", False) is True else False,
            "qtranform_runid": ID,
            
        }, f=checkpoint_path)
    log.info(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path