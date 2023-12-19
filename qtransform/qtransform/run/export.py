
import logging
from typing import Any
from omegaconf import DictConfig
import hydra
import os
from qtransform.utils.helper import load_checkpoint
import torch
from brevitas.export import export_onnx_qcdq, export_qonnx, export_brevitas_onnx
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

    log.trace(f"Model structure: {model}")
    log.debug(f"Model config from checkpoint: {checkpoint['model_cfg']}")

    input_dim = (1, checkpoint['model_cfg']['args']['block_size'])
    max_token_id = checkpoint['model_cfg']['args']['vocab_size']
    sample_tensor = torch.randint(0, max_token_id, input_dim, dtype=int)

    filename = cfg.run.from_checkpoint.split("/")[-1] + ".onnx"
    if cfg.run.get("output"):
        filename = cfg.run.get("output")

    """
    export function maps to torch onnx export:
    # Export the model
        torch.onnx.export(torch_model,  # model being run
        x,                              # model input (or a tuple for multiple inputs)
        "model.onnx",        # where to save the model (can be a file or file-like object)
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=10,               # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names = ['input'],        # the model's input names
        output_names = ['output'],      # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}})
    """
    kwargs = {
        "input_names" :['input', 'offsets'],   # the model's input names
        "output_names" : ['output'],         # the model's output names
        "dynamic_axes" :{'input' : {0 : 'batch_size'},    # variable length axes
                        'output' : {0 : 'batch_size'}},
        "opset_version": cfg.run.opset_version,  
        "export_params": True,  
    }
    if cfg.run.export_fn == "export_qonnx":
        try:
            export_qonnx(model, torch.tensor(sample_tensor), export_path=f"qonnx_{str(input_dim)}_" + filename, **kwargs)
        except Exception:
            log.error(f"Export via {export_qonnx.__module__}.{export_qonnx.__name__} failed, reason", exc_info=True)
   
    if cfg.run.export_fn == "export_onnx_qcdq":             
        try:
            export_onnx_qcdq(model, torch.tensor(sample_tensor), export_path=f"onnx_qcdq_{str(input_dim)}_" + filename, **kwargs)
        except:
            log.error(f"Export via {export_onnx_qcdq.__module__}.{export_onnx_qcdq.__name__} failed, reason", exc_info=True)

    if cfg.run.export_fn == "export":           
        try:
            export(model, torch.tensor(sample_tensor), f"onnx_{str(input_dim)}-" + filename, opset_version=opset_version,  export_params=True, **kwargs)
        except:
            log.error(f"Export via {export.__module__}.{export.__name__} failed, reason", exc_info=True)
