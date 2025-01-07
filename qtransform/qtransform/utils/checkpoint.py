import datetime
import os
from typing import Any, Dict, Tuple, Union, TypedDict
import torch
import logging
from dataclasses import dataclass, fields, is_dataclass, field
from torch import nn
from torch.optim import Optimizer
from pprint import PrettyPrinter
from qtransform import ConfigSingleton
from qtransform.utils import ID
from qtransform.utils.helper import get_output_chkpt_dir
from qtransform import device_singleton
log = logging.getLogger(__name__)

def get_from_config_f(param: str): 
    def get_from_config_inner():
        from hydra.core.hydra_config import HydraConfig
        return HydraConfig.get(param)
    return get_from_config_inner

@dataclass
class QtransChkptMetaData():
    """ torch checkpoint meta data, also a key mapping for omegaconf"""
    # epoch: int
    # steps: int
    # metrics: Any
    # run_id: str
    # timestamp: datetime
    # qtrans_model_config: Any
    # qtrans_dataset_config: Any
    # qtrans_quantization_config: Any
    # qtrans_tokenizer_config: Any
    
    member1: int = field(default_factory=get_from_config_f("test"))
    member2: str = field(default="Standardwert")
    member3: float = field(default=1.0)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QtransChkptMetaData':
        return cls(**{k: data.get(k, field.default) for k, field in cls.__dataclass_fields__.items()})
        
def validate_checkpoint(checkpoint: Dict):
    """ checks if all necessary meta data is present in checkpoint
    and return False if that is not the case.
    """


    return True


def load_checkpoint(checkpoint_path: str):
    """ load torch model checkpoint and return the epoch count"""
    if not os.path.exists(checkpoint_path):
        log.error(f'Checkpoint {checkpoint_path} does not exist')
        raise FileNotFoundError()
    log.info(f"Loading checkpoint from {checkpoint_path}")   
    checkpoint = torch.load(checkpoint_path, map_location=device_singleton.device)
    return checkpoint

def load_state_dict_proxy(model, checkpoint, **kwargs):
    """same as torch load state dict, however this check env for extra params. """
    strict = bool(int(os.environ.get("qtransform_load_state_strict", 1)))
    if "strict" not in kwargs.keys():
        kwargs.update({"strict": strict})
    return model.load_state_dict(checkpoint, **kwargs)

def save_checkpoint(model: nn.Module, 
    optimizer: Optimizer,
    metadata: QtransChkptMetaData = None,
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
        filename = f"{cfg.model.get('model_name')}_{dataset_name.replace('/', '__')}_{metadata.timestamp}__epoch:{metadata.epoch}"
    else:
        filename = f"{cfg.runtime.choices.model}_{dataset_name.replace('/', '__')}_{metadata.timestamp}__epoch:{metadata.epoch}"

    checkpoint_path = os.path.join(get_output_chkpt_dir(), filename)
    log.info(f"Model checkpoint saving to {checkpoint_path}")
    torch.save(obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "qtrans_metadata": metadata
            
        }, f=checkpoint_path)
    log.info(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path
