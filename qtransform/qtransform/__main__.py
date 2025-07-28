import os
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from dataclasses import asdict
import logging
import qtransform
from qtransform.utils import addLoggingHandler, addLoggingLevel
from qtransform import ConfigSingleton, device_singleton
import torch
from qtransform.utils.helper import get_output_log_dir
import sys
import json
import wandb
from qtransform.utils.id import ID
from qtransform.wandb import wandb_init
from qtransform.utils.checkpoint import load_checkpoint
from qtransform.utils.checkpoint import QtransChkptMetaData

addLoggingLevel("TRACE", logging.DEBUG - 5, "trace")
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=qtransform.get_module_config_path(), config_name="config.yaml")
def cli_wrapper(cfg: DictConfig):
    """ 
    this function exsists so that one can call qtransform from cli with prepending "python -m ".
    note that additional configs can be loaded via --config-dir https://github.com/facebookresearch/hydra/issues/874
    """
    ConfigSingleton().config = cfg
    addLoggingHandler(os.path.join(get_output_log_dir() , ID + ".log"))
    main()

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def module_wrapper(cfg: DictConfig):
    ConfigSingleton().config = cfg
    addLoggingHandler(os.path.join(get_output_log_dir() , ID + ".log"))
    main()
    

def main():
    # TODO do we nned check if execution is inside slurm?
    # if 'SLURM_JOB_ID' not in os.environ:
    #     print("This script is not running inside a SLURM job.")
    # else:
    #     print(f"This script is running inside a SLURM job with SLURM_JOB_ID={os.environ['SLURM_JOB_ID']}")
    
    cfg = ConfigSingleton().config
    OmegaConf.update(cfg, "runtime.choices", HydraConfig().instance().get().runtime.choices, force_add=True)
    logging.captureWarnings(True)
    root_log = logging.getLogger("root")
    log = logging.getLogger(f"{__package__}.{__name__}")
    
    if "trace" in cfg and cfg.trace:
        root_log.setLevel(logging.TRACE)
        log.warning("TRACE ENABLED")
    elif "debug" in cfg and cfg.debug:
        root_log.setLevel(logging.DEBUG)
        log.debug("DEBUG ENABLED")
    
    if hasattr(log, "trace"): log.trace("launched with config: " + json.dumps(OmegaConf.to_container(cfg), indent=2))
    log.info(f"Launch command: {sys.argv}")
    log.info(f"qtransform ID: {ID}")
    hydra_cli_args = OmegaConf.to_container(HydraConfig.get().overrides.task)
    log.info(f"{hydra_cli_args=}")
    log.info(f'LAUNCHED WITH CONFIG: {cfg}')
    
    if "device" not in cfg:
        device_singleton.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info(f"device not specified in config, using default device: {device_singleton.device}")
    else:
        device_singleton.device = cfg.device
        device = device_singleton.device
    if "seed" not in cfg:
        # generate a random seed
        cfg.seed = torch.randint(0, 2**32 - 1, (1,)).item()
        log.info(f"seed not specified in config, using default seed: {cfg.seed}")
    else:
        log.info(f"using seed: {cfg.seed} from config")
        torch.manual_seed(cfg.seed)    
    
    # before we start wandb we need to read out the checkpoint if we continue training
    checkpoint_location = cfg.get('model')
    checkpoint_location = checkpoint_location.get('checkpoint')
    if checkpoint_location is not None:
        # find overwritten arguments
        log.info(f"checkpoint {checkpoint_location} was given, loading meta data to continue from")
        checkpoint = load_checkpoint(checkpoint_location)
        for key, value in checkpoint.items():
            log.info(f"checkpoint key: {key}")
        checkpoint_metadata = checkpoint.get('qtrans_metadata')
        
        if checkpoint_metadata is None:
            # raise ValueError(f"checkpoint {checkpoint_location} does not contain qtransform metadata")
            log.warning(f"checkpoint {checkpoint_location} does not contain qtransform metadata, using default values or values from command line")
        elif not isinstance(checkpoint_metadata, QtransChkptMetaData): # checkpoint metadata cloud be a dict or the class via pickle
            checkpoint_metadata = QtransChkptMetaData.from_dict(checkpoint_metadata)
            log.info(f"Overwritten arguments from checkpoint: {checkpoint_metadata.qtrans_hydra_overrides}")
        
        def merge_hydra_cli_args(checkpoint_overrides, cli_overrides):
            """
            Merges the CLI overrides with the checkpoint overrides.
            CLI overrides take precedence over checkpoint overrides.
            """
            checkpoint_overrides_dict = {item.split('=')[0]: item.split('=')[1] for item in checkpoint_overrides}
            cli_overrides_dict = {item.split('=')[0]: item.split('=')[1] for item in cli_overrides}
            merged_overrides = {**checkpoint_overrides_dict, **cli_overrides_dict}
            return [f"{key}={value}" for key, value in merged_overrides.items()]
        
        if checkpoint_metadata is not None: # if checkpoint metadata cloud not be loaded, we use the default values
            merged_overrides_array = merge_hydra_cli_args(checkpoint_metadata.qtrans_hydra_overrides, hydra_cli_args)
        else:
            # some runttime arguments cloud be missing here so procuede carefully with the cli args
            merged_overrides_array = hydra_cli_args
            
        log.info(f"merged overwrites: {merged_overrides_array}")
        cfg = hydra.compose("config", overrides=merged_overrides_array)
        # cfg = hydra.compose(overrides=hydra_cli_args)
        OmegaConf.update(cfg, "runtime.overwrites", merged_overrides_array, force_add=True)
        ConfigSingleton().config = cfg
        log.info(cfg)

    # start programm now...
    exit_code=0
    log.info(cfg)
    wandb_init(cfg) 
    if "command" not in cfg.run:
        log.error("No run command found in run config, run config was: " + str(cfg.run))
        raise KeyError
    try:
        # dynamicly import run module from qtransform.run
        module = __import__("qtransform.run", fromlist=[cfg.run.command])
        cmd = getattr(module, cfg.run.command)
        cmd.run(cfg)
    except Exception as e:
        exit_code = 1 # generic error
        log.critical("Script execution failed. Reason: ", exc_info=True)
        
        # OutOfMemoryError does not exsist in static python torch package...
        if e.__class__.__name__ == "OutOfMemoryError":
            exit_code = 2

    # make sure wandb is closed (gets checked internally wether wandb.run is activ) 
    wandb.finish(exit_code=exit_code)

    if exit_code is not None and exit_code > 0:
        log.error(f"Exited with Status Code: {str(exit_code)}")
        exit(exit_code)
    log.info(f"Exited with Status Code: {str(exit_code)}")

if __name__ == "__main__":
    module_wrapper()