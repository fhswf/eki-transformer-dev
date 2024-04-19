import datetime
import os
from typing import Any, Dict, Tuple, Union
import hydra
from pprint import pprint
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf, DictConfig, open_dict
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
from hydra.core.utils import JobReturn, JobStatus
from hydra.experimental.callback import Callback
from pathlib import Path
import pickle
from pprint import PrettyPrinter

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
    assert isinstance(filename, str), f'Could not load checkpoint/model with filename: "{filename}"'
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


#overall, callbacks arent that useful for manipulating the state of the application, but they could be useful for type checking
#the problem is that callbacks are called with the config at the start of the run script and do not reflect changes made to it
#it is possible to define our own callback interface and use them within the main script
#it does the same thing but it retrieves config via HydraConfig inbetween each event, thereby allowing changes
#it is not as configurable though
class FromFileInfoCallback(Callback):
    "Updates the model.from_file field to suit the newly generated checkpoint/ onnx model"

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        #######################
        from_previous_run = config.from_previous_run
        if from_previous_run is None:
            return
        log.info(f'Updating config with from_previous_run={from_previous_run}')
        if os.path.isfile(from_previous_run):
            log.warn(f'from_previous_run expects directory path, not filepath. Removing filename')
            output_dir, _ = os.path.split(from_previous_run)
        elif os.path.isabs(from_previous_run):
            output_dir = from_previous_run
        else:
            #remove current timestamp
            output_dir, _ = os.path.split(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            output_dir = os.path.join(output_dir, from_previous_run, '.hydra')
        config_path = os.path.join(output_dir, 'config.pickle')
        with open(config_path, 'rb') as input:
            new_config = pickle.load(input)
        assert isinstance(config, DictConfig), f'Pickle file from {from_previous_run} is not a DictConfig file'
            #del config["run"]
        log.debug(f'Loaded config from previous run: {PrettyPrinter(indent=1).pformat(config)}')
        #update global hydra config to access new keys in callbacks
        #hydra.runtime.choices: chosen yaml cfg files
        #hydra.overrides.task: fields overwritten in cmd line
        current_cfg = HydraConfig.instance().get()
        """
        with open_dict(new_config):
            #for now, ignore overrides from cli except for "run" and start with previous cfg
            new_overrides = set(new_config.hydra.overrides.task) - set(current_cfg.overrides.task)
            log.warning(f'Overrides for {new_overrides} except for "run" are ignored')
            #which yaml files for each config group were chosen
            new_config.hydra.runtime.choices = current_cfg.runtime.choices
            new_config.hydra.runtime.cwd = current_cfg.runtime.cwd
            new_config.hydra.runtime.output_dir = current_cfg.runtime.output_dir
            new_config.hydra.job.override_dirname = current_cfg.job.override_dirname
        """
        #problem: jobs are supplied with the same config within function run_job
        #https://github.com/facebookresearch/hydra/blob/main/hydra/core/utils.py
        #we could fork hydra and do something about it but that wouldnt be a good idea
        HydraConfig.instance().set(new_config)

    def on_job_end(self, config: DictConfig, **kwargs: Any) -> None:
        #cfg.model.from_file = "TEST"
        pass

#from: hydra.experimental.callbacks
class PickleJobInfoCallback(Callback):
    """Pickle the job config/return-value in ${output_dir}/{config,job_return}.pickle"""

    output_dir: Path

    def __init__(self) -> None:
        self.log = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_job_start(self, config: DictConfig, **kwargs: Any) -> None:
        log.info(f'Saving hydra config at the end of run script')

    def on_job_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Pickle the job's config in ${output_dir}/config.pickle.
        It is saved at the end in order to reflect dynamic changes in the config
        """
        #problem: if from_previous_run is supplied, config fields are not updated when pickling them
        self.output_dir = Path(config.hydra.runtime.output_dir) / Path(
            config.hydra.output_subdir
        )
        filename = "config.pickle"
        #TODO: update model.from_file in HydraConf.instance()
        self._save_pickle(obj=config, filename=filename, output_dir=self.output_dir)
        self.log.info(f"Saving job configs in {self.output_dir / filename}")

        #Pickle the job's return value in ${output_dir}/job_return.pickle.
        filename = "job_return.pickle"
        self._save_pickle(obj=job_return, filename=filename, output_dir=self.output_dir)
        self.log.info(f"Saving job_return in {self.output_dir / filename}")

    def _save_pickle(self, obj: Any, filename: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir is not None
        with open(str(output_dir / filename), "wb") as file:
            pickle.dump(obj, file, protocol=4)