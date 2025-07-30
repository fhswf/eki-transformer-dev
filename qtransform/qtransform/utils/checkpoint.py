import datetime
import os
from typing import Any, Dict, Tuple, Union, TypedDict, Optional
import torch
import logging
from dataclasses import dataclass, fields, is_dataclass, field
from torch import nn
from torch.optim import Optimizer
from pprint import PrettyPrinter
from qtransform import ConfigSingleton
from qtransform.utils.id import ID
import wandb
from qtransform.utils.helper import get_output_chkpt_dir
from qtransform import device_singleton
from hydra.core.hydra_config import HydraConfig
log = logging.getLogger(__name__)

def get_from_config_f(param: str, default: Any = None): 
    def get_from_config_inner():
        if default is not None and callable(default):
            return HydraConfig.get().get(param, default())
        else:
            return HydraConfig.get().get(param, default)
    return get_from_config_inner

@dataclass
class QtransChkptMetaData():
    """ torch checkpoint meta data, also a key mapping for omegaconf"""
    metrics: Optional[Any] =  field(default=None)
    epoch: int = field(default=1)
    steps: int = field(default=0)
    run_id: str = field(default_factory=lambda:ID)
    model_name: str = field(default_factory=get_from_config_f("model.model_name", lambda: ID+(HydraConfig.get()["runtime"]["choices"]["model"] or "")+(HydraConfig.get()["runtime"]["choices"]["dataset"] or "")))
    qtrans_hydra_overrides: Dict[str, Any] = field(default_factory=lambda: HydraConfig.get().overrides.task)
    checkpoint_counter: int = field(default=0)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QtransChkptMetaData':
        return cls(**{k: data.get(k, field.default) for k, field in cls.__dataclass_fields__.items()})
    
    def to_dict(self) -> Dict[str, Any]:
        return {field.name: getattr(self, field.name) for field in fields(self)}
    
def load_checkpoint(checkpoint_path: str):
    """ load torch model checkpoint and return the epoch count. Note that torch.load is using python pickle magic."""
    if not os.path.exists(checkpoint_path):
        log.error(f'Checkpoint {checkpoint_path} does not exist')
        raise FileNotFoundError()
    log.info(f"Loading checkpoint from {checkpoint_path}")   
    print(device_singleton.device)
    checkpoint = torch.load(checkpoint_path, map_location=device_singleton.device)
    return checkpoint

def load_state_dict_proxy(model, checkpoint, **kwargs):
    """same as torch load state dict, however this check env for extra params. """
    strict = bool(int(os.environ.get("QTRANSFORM_LOAD_STATE_STRICT", 1)))
    if "strict" not in kwargs.keys():
        kwargs.update({"strict": strict})
    return model.load_state_dict(checkpoint, **kwargs)

def save_checkpoint(model: nn.Module, 
    optimizer: Optimizer,
    **kwargs) -> str:
    """save torch model checkpoint from training, returns path to saved file."""
    metadata: QtransChkptMetaData = None
    # TODO this is not working, the metadata is not passed
    if "metadata" == kwargs.keys() and type(kwargs["metadata"]) is QtransChkptMetaData:
        metadata = kwargs["metadata"]
    else:
        metadata = QtransChkptMetaData(**kwargs)
        
    # Ensure metadata contains qtrans_ attributes, if not fill them from hydra config
    if not metadata.qtrans_hydra_overrides:
        metadata.qtrans_hydra_overrides = HydraConfig.get().overrides.task
        
    # Check if ConfigSingleton contains "runtime.overrides" and update qtrans_hydra_overrides
    if "runtime" in ConfigSingleton().config and "overwrites" in ConfigSingleton().config["runtime"]:
        metadata.qtrans_hydra_overrides = ConfigSingleton().config["runtime"]["overwrites"]
    
    metadata.checkpoint_counter = metadata.checkpoint_counter + 1
    checkpoint_path = os.path.join(get_output_chkpt_dir(), metadata.model_name)
    log.info(f"Model checkpoint saving to {checkpoint_path}")
    torch.save(obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "qtrans_metadata": metadata
        }, f=checkpoint_path)
    log.info(f"Checkpoint counter: {metadata.checkpoint_counter}")
    log.info(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path
