
import torch

## Weight-only model
#export_onnx_qcdq(model, torch.randn(1, 3, 32, 32), export_path='4b_weight_lenet.onnx')
#
## Weight-activation model
#export_onnx_qcdq(model, torch.randn(1, 3, 32, 32), export_path='4b_weight_act_lenet.onnx')
#
## Weight-activation-bias model
#export_onnx_qcdq(model, torch.randn(1, 3, 32, 32), export_path='4b_weight_act_bias_lenet.onnx')


import logging
from typing import Any
from omegaconf import DictConfig
import hydra
import os
import torch
from brevitas.export import export_onnx_qcdq
from datetime import datetime

log = logging.getLogger(__name__)

def run(cfg: DictConfig):
    """ exports a trained model to QONNX or others?"""
    log.info("================")
    log.info("Exporting Model")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")
    
    pass