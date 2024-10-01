from dataclasses import dataclass

from copy import deepcopy
from typing import Any
import numpy as np
from omegaconf import DictConfig, open_dict
import hydra
import os
from qtransform import device_singleton
from qtransform.utils.helper import get_output_dir, load_checkpoint
from qtransform.model import ModelArgs, GenericModel, get_model_wrapper, DynamicCheckpointQTRModelWrapper
from qtransform.utils.id import ID
import torch
from torch import nn
from brevitas.export import export_onnx_qcdq, export_qonnx, export_brevitas_onnx
from brevitas import nn as qnn
from torch.onnx import export
from datetime import datetime
import matplotlib.pyplot as plt


import logging
log = logging.getLogger(__name__)

@dataclass
class KwargsAnalysisConfig():
    """
    Note that this is not a class for holding a hydra config dict.
    This dataclass represents kwargs options for this modules run command.
    KwargsAnalysisConfig should only be used when actually chaining commands together.
    """
    #find_outliers: bool = False
    #find_device_mismatch: bool = False
    #find_untrained_quantizers: bool = False
    pass

def run(cfg : DictConfig, **kwargs):
    """ Analysis module """
    log.info("================")
    log.info("Running Analysis")
    log.info("================")
    log.info(cfg)
    # global device might be bad, as we want to make sure device switching works
    device_singleton.device = cfg.device
    device = device_singleton.device

    # TODO get model from kwargs if applicable
    # TODO get other intormation from kwargs if applicable
    # what we need to do here depends on analysis steps

    model_wrapper: DynamicCheckpointQTRModelWrapper = get_model_wrapper(cfg.model)
    model: GenericModel = model_wrapper.model
    assert isinstance(model, GenericModel), f'model is not of type GenericModel'
    #log.debug(f"Model structure: {model}") 

    if cfg.run.find_outliers:
        find_outliers(cfg, model)
        
def find_untrained_quantizers(cfg, model):
    pass

@torch.no_grad()
def find_outliers(cfg, model: nn.Module):
    log.info(f"{len(list(model.named_parameters()))=}")
    log.info(f"{len(list(model.parameters()))=}")
    if not len(list(model.parameters())) == len(list(model.named_parameters())):
        log.warning(f"not all parameters in this model are named, this can lead to some missing results")
    
    for name, param in model.named_parameters():
    
        if (len(list(param.data.shape)) != 0):
           
            std_div = np.std(param.data.numpy(force=True))
            mean = np.mean(param.data.numpy(force=True))
            diff = np.abs((param.data.numpy(force=True) - mean)/std_div)
            if np.max(diff) > 3.3:
                #print("===============")
                log.warning(f"{(diff > 3.3).sum()} outlier up to {np.max(diff):2.6f} in layer {name} with {std_div=:2.6f}, {mean=:2.6f}")
                #print(param.data)
                #print(diff)
                plt.cla()
                _diff = diff
                #print(diff.shape)
                if len(diff.shape) > 2:
                    for i in range(0, len(diff.shape) - 2):
                        _diff = np.squeeze(diff,0)
                    plt.imshow(_diff, interpolation='none')
                elif len(diff.shape) == 1:   
                    plt.figure(figsize=(12, 12))
                    plt.bar(range(0,diff.shape[0]),_diff)
                    #plt.plot(_diff)
                else:
                    plt.imshow(_diff, interpolation='none')
                os.makedirs(os.path.join(get_output_dir(), "analysis") , exist_ok=True)
                plt.savefig(os.path.join(get_output_dir(), "analysis", f'{ID}-{name}.png'))
                #print("===============")
    pass