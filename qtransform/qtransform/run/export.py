
import logging
from typing import Any
from omegaconf import DictConfig
import hydra
import os
from qtransform.utils.helper import load_checkpoint
import torch
from brevitas.export import export_onnx_qcdq
from torch.onnx import export
from datetime import datetime
import torch


log = logging.getLogger(__name__)

def run(cfg: DictConfig):
    """ exports a trained model to QONNX or others?"""
    log.info("================")
    log.info("Exporting Model")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")
    
    # load model checkpoint
    _, checkpoint = load_checkpoint(cfg=cfg)
    print(checkpoint.keys())

    from qtransform.model import get_model
    model = None
    if "model" in cfg and "cls" in cfg.model:
        model = get_model(cfg.model)
    elif "model_cfg" in checkpoint:
        model = get_model(checkpoint['model_cfg'])
    else:
        log.error("No model defintion provided in either checkpoint or cfg.model")
        return 1
    model.load_state_dict(checkpoint['model_state_dict'])
    """
    export function maps to torch onnx export:
    # Export the model
        torch.onnx.export(torch_model,  # model being run
        x,                              # model input (or a tuple for multiple inputs)
        "super_resolution.onnx",        # where to save the model (can be a file or file-like object)
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=10,               # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names = ['input'],        # the model's input names
        output_names = ['output'],      # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}})
    
    """

    log.info(f"Model structure: {model}")
    if cfg.run.get("output"):
        # if not quant:
        export(model, torch.tensor([[1337, 420, 360]]), cfg.run.get("output"), opset_version=16)
        #else:
        export_onnx_qcdq(model, torch.tensor([[1337, 420, 360]]), export_path=cfg.run.get("output"), opset_version=16)
    else:
        filename = cfg.run.from_checkpoint.split("/")[-1]
        # if not quant:
        export(model, torch.tensor([[1337, 420, 360]]), filename, opset_version=16)
        #else:
        export_onnx_qcdq(model, torch.tensor([[1337, 420, 360]]), "q" + filename, opset_version=16)

       
    # Weight-only model

 

    pass