import datetime
import os
from typing import Any, Dict, Tuple, Union
import hydra
from omegaconf import DictConfig
import torch
import logging
from torch import nn
from stat import S_ISFIFO
log = logging.getLogger(__name__)



def get_default_chkpt_folder() -> str:
    """
        Returns the default directory where model checkpoints are stored if the path was not configured
        by the user with the cfg variables "checkpoint_dir".
    """
    return os.path.join(os.getenv("HOME"), *__package__.split("."), "checkpoint_dir")

def load_checkpoint(cfg: DictConfig) -> Tuple[int, Union[Any, Dict]]:
    """ load torch model checkpoint"""
    #checkpoint_dir: specify custom path for checkpoints, otherwise use directory checkpoint_dir in qtransform/utils/checkpoint_dir
    chkpt_folder = get_default_chkpt_folder()
    if "from_checkpoint" not in cfg.run:
        log.error(f'Key "from_checkpoint" not specified in run config, e.g. run.from_checkpoint="<path>"')
        raise KeyError()
    #from_checkpoint is the absolute path to a file, ignore checkpoint_dir 
    if os.path.isabs(cfg.run.from_checkpoint):
        chkpt_folder, from_checkpoint = os.path.split(cfg.run.from_checkpoint)
    elif "checkpoint_dir" in cfg.run:
        if os.path.isabs(cfg.run.checkpoint_dir):
            chkpt_folder = cfg.run.checkpoint_dir
            from_checkpoint = cfg.run.from_checkpoint
        else:
            #outputs are stored in <current directory>/outputs/<checkpoint_dir>
            try:
                chkpt_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, "outputs", cfg.run.checkpoint_dir)
            except:
                log.debug(f'Could not get cwd from hydra. Reason: ', exc_info=True)
                log.debug(f'Using os.getcwd')
                chkpt_folder = os.getcwd()
            from_checkpoint = cfg.run.from_checkpoint
    checkpoint_path = os.path.join(chkpt_folder, from_checkpoint)
    if not os.path.isfile(checkpoint_path):
            log.error(f"Checkpoint {checkpoint_path} is not a file")
            raise FileNotFoundError
    log.info(f"Loading checkpoint from {checkpoint_path}")
    from_epoch = 0    
    from qtransform import device_singleton
    checkpoint = torch.load(checkpoint_path, map_location=device_singleton.device)
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
    tokenizer_cfg: Any,
    quant_cfg: Union[DictConfig, None] = None) -> str:
    """save torch model checkpoint from training, returns path to saved file."""
    
    chkpt_folder = get_default_chkpt_folder()
    if "checkpoint_dir" in cfg.run:
        if os.path.isabs(cfg.run.checkpoint_dir):
            chkpt_folder = cfg.run.checkpoint_dir
        else:
            try:
                chkpt_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, "outputs", cfg.run.checkpoint_dir)
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
                "quant_cfg": quant_cfg, 
                "quantized": True if cfg.get('quantization', {"quantize": False})["quantize"] is True else False
                }, f=checkpoint_path)
        log.info(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path

import onnx
import onnxruntime as ort
from qonnx.core.modelwrapper import ModelWrapper
# maybe only do this when it is required, for this howiever is always the case
from onnx.shape_inference import infer_shapes

def load_onnx_model(path: str) -> ModelWrapper:
    """
    Loads ONNX model from a filepath. 
    """
    if not isinstance(path, str):
        log.error(f'Could not load ONNX model because: path {path} is not a string.')
    if not os.path.isfile(path):
        log.error(f'Could not load ONNX model because: path {path} is not a file.')
    log.info(f'Loading ONNX model from "{path}"')
    #qonnx lib also works with onnx models
    model = ModelWrapper(path)    
    infered_shapes = infer_shapes(model.model)
    return ModelWrapper(infered_shapes)

def write_to_pipe(cfg: DictConfig, content: str) -> None:
    """
    Write into a named pipe in order for other qtransform processes to access information from this current instance.
    The filepath of checkpoints or ONNX models could be written into the pipe in order to continue training from it or 
    perform inference or benchmarking. 

    The "pipe" field from the hydra config specifies the filepath of the named pipe (by default: /dev/null). If the pipe does
    not exist yet and the current operating system is UNIX-like, it will be created. 
    """
    #write checkpoint into named_pipe
    pipe_name = cfg.get('pipe', '/dev/null')
    #by default, write checkpoint to /dev/null if pipe name is omited
    pipe_name = '/dev/null' if pipe_name is None else pipe_name
    if not os.path.exists(pipe_name):
        from sys import platform
        if platform == "win32":
            log.error(f'Cannot create pipes on non-UNIX system.')
            raise RuntimeError()
        log.debug(f'Creating named pipe "{pipe_name}"')
        os.mkfifo(pipe_name)

    if pipe_name == '/dev/null': #avoid logging when output goes nowhere
        return
    elif not S_ISFIFO(os.stat(pipe_name).st_mode):
        log.error(f'Specified filepath "{pipe_name}" is not a pipe.')
    else:
        log.info(f'Writing content "{content}" into fifo "{pipe_name}". ' \
                 f'Until another process reads from the fifo, the current process (PID {os.getpid()}) is blocked.'
        )
        #writing into a named pipe blocks process until another process reads from it
        #problematic if something should be done after writing into the pipe
        with open(pipe_name, 'w') as pipe:
            pipe.write(content)