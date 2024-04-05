import datetime
import os
from typing import Any, Dict, Tuple, Union
import hydra
from pprint import pprint
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig
import torch
import logging
from torch import nn
from stat import S_ISFIFO
from yaml import dump, safe_load
import onnx
import onnxruntime as ort
from qonnx.core.modelwrapper import ModelWrapper
# maybe only do this when it is required, for this howiever is always the case
from onnx.shape_inference import infer_shapes
from dataclasses import dataclass

log = logging.getLogger(__name__)

@dataclass
class FromFile():
    """
    Keep args for composing checkpoint/ onnx model path inside of dataclass to avoid dict checks in multiple
    places.
    """
    filename: str
    model_dir: str = None

def from_meta(meta_file: str) -> DictConfig:
    pass

def generate_meta(cfg: DictConfig, model: Union[torch.nn.Module, ModelWrapper]) -> str:
    """
    Generates meta from hydra config to be used in future runs.
    With a meta file, no further configs need to be specified in each run script.  

    Args: cfg: The hydra config
    Returns: The filepath to the meta file
    """
    #dataset
    #tokenizer
    #model
    #quant_cfg
    dump()

def get_default_chkpt_folder() -> str:
    """
        Returns the default directory where model checkpoints are stored if the path was not configured
        by the user with the cfg variables "checkpoint_dir".
    """
    return os.path.join(os.getenv("HOME"), *__package__.split("."), "checkpoint_dir")

def compose_model_path(from_file: FromFile, not_exist_error = True) -> str:
    """
    Composes an absolute filepath from a filename and a directory path. 
    The filename can be absolute, ignoring model_dir. Otherwise, model_dir is prepended to filename.
    """
    if not isinstance(from_file, FromFile):
        if isinstance(from_file, Union[Dict, DictConfig]):
            from_file = FromFile(**from_file)
        else:
            log.error(f'Cannot compose model_path with from_file: {from_file}')
            raise TypeError()
    filename = from_file.filename
    model_dir = from_file.model_dir
    chkpt_folder = get_default_chkpt_folder()
    #from_checkpoint is the absolute path to a file, ignore checkpoint_dir 
    if os.path.isabs(filename):
        chkpt_folder, from_checkpoint = os.path.split(filename)
    elif isinstance(model_dir, str):
        if os.path.isabs(model_dir):
            chkpt_folder = model_dir
            from_checkpoint = filename
        else:
            #outputs are stored in <current directory>/outputs/<checkpoint_dir>
            try:
                chkpt_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, "outputs", model_dir)
            except:
                log.debug(f'Could not get cwd from hydra. Reason: ', exc_info=True)
                log.debug(f'Using os.getcwd')
                chkpt_folder = os.getcwd()
            from_checkpoint = filename
    else:
        log.warning(f'Directory to model "{filename}" omited. Assuming default "{chkpt_folder}"')
        from_checkpoint = filename
        #log.error(f'Path to model could not be resolved with filename: "{filename}", model_dir: "{model_dir}" '\
        #    f'Probably because the filename was not an absolute path and model_dir was not specified.')
        #raise RuntimeError()
    checkpoint_path = os.path.join(chkpt_folder, from_checkpoint)
    #if filename should exist (not_exist_error == True), throw error
    if not os.path.isfile(checkpoint_path) and not_exist_error:
            log.error(f"Checkpoint {checkpoint_path} is not a file")
            raise FileNotFoundError
    return checkpoint_path

def load_checkpoint(from_file: Union[Dict, DictConfig, FromFile]) -> Tuple[int, Union[Any, Dict]]:
    """ load torch model checkpoint and return the epoch count"""
    checkpoint_path = compose_model_path(from_file)
    log.info(f"Loading checkpoint from {checkpoint_path}")
    from_epoch = 0    
    from qtransform import device_singleton
    checkpoint = torch.load(checkpoint_path, map_location=device_singleton.device)
    if 'epoch' in checkpoint:
        from_epoch = checkpoint['epoch']
    else:
        try:
            i = str(filepath).index("epoch:")
            import re
            p = re.compile("[0-9]+")
            from_epoch = int(p.search(str(cfg.run.from_checkpoint)[i:]).group(0))
        except ValueError or AttributeError:
            log.warn("Modelcheckpint does not contain epoch information")
    return from_epoch,checkpoint

def load_state_dict_proxy(model, checkpoint, **kwargs):
    """same as torch load state dict, however this check env for extra params. """
    strict = bool(int(os.environ.get("qtransform_load_state_strict", 1)))
    if "strict" not in kwargs.keys():
        kwargs.update({"strict": strict})
    return model.load_state_dict(checkpoint, **kwargs)

def save_checkpoint(from_file: Union[Dict, DictConfig, FromFile],
    model: nn.Module, 
    dataset_name: str, 
    optimizer, 
    timestamp:datetime, 
    metrics:Dict, 
    epoch:int, 
    model_cfg: Any,
    tokenizer_cfg: Any,
    quant_cfg: Union[DictConfig, None] = None) -> str:
    """save torch model checkpoint from training, returns path to saved file."""
    

    if not isinstance(from_file, FromFile) and isinstance(from_file, Union[Dict, DictConfig]):
        from_file = FromFile(**from_file)
    else:
        log.error(f'Cannot compose model_path with from_file: {from_file}')
        raise TypeError()
    from_file.filename = f"{OmegaConf.to_container(HydraConfig.get().runtime.choices)['model']}_{dataset_name.replace('/', '__')}_{timestamp}__epoch:{epoch}"
    checkpoint_path = compose_model_path(from_file, not_exist_error=False)
    chkpt_folder, _ = os.path.split(checkpoint_path)
    os.makedirs(chkpt_folder, exist_ok=True)
    log.info(f"Model checkpoint saving to {checkpoint_path}")
    torch.save(obj={
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "model_cfg": model_cfg,
            "tokenizer_cfg": tokenizer_cfg, 
            "metrics": metrics,
            "quant_cfg": quant_cfg, 
            "quantized": True if quant_cfg.get("quantize", False) is True else False
            }, f=checkpoint_path)
    log.info(f"Model checkpoint saved to {checkpoint_path}")
    return checkpoint_path


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