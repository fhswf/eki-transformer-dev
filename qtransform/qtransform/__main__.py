import os
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose
from omegaconf import DictConfig, OmegaConf
from dataclasses import asdict
import logging
import qtransform
from qtransform.utils import addLoggingHandler, addLoggingLevel
from qtransform import ConfigSingleton
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
    
def find_overwritten_args_from_cli(cfg, cli_args):
    diff = {}
    for arg in cli_args:
        if '=' in arg:
            key, value = arg.split('=', 1)
            keys = key.split('.')
            current_cfg = cfg
            for k in keys[:-1]:
                current_cfg = current_cfg.get(k, {})
            if keys[-1] in current_cfg and str(current_cfg[keys[-1]]) != value:
                diff[key] = current_cfg[keys[-1]]
    return diff

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

    log.debug(f'LAUNCHED WITH CONFIG: {cfg}')
    
    # before we start wandb we need to read out the checkpoint if we continue training
    checkpoint_location = cfg.get('model')
    checkpoint_location = checkpoint_location.get('checkpoint')
    if checkpoint_location is not None:
        # find overwritten arguments
        cli_args = sys.argv[1:]  # Exclude the script name
        log.info(f"Overwritten arguments: {cli_args}")
        log.info(f"checkpoint {checkpoint_location} was given, loading meta data to continue from")
        checkpoint = load_checkpoint(checkpoint_location)
        checkpoint_metadata = checkpoint.get('qtrans_metadata')
        if checkpoint_metadata is None:
            raise ValueError(f"checkpoint {checkpoint_location} does not contain qtransform metadata")
        if not isinstance(checkpoint_metadata, QtransChkptMetaData):
            checkpoint_metadata = QtransChkptMetaData.from_dict(checkpoint_metadata)
        
        # Update the current config with the checkpoint metadata, only if the current config values are missing
        def update_config(cfg, metadata, overwritten_args):
            updated_values = {}
            for key, value in metadata.items():
                if key.startswith("qtrans_") and key.endswith("_config"):
                    stripped_key = key[len("qtrans_"):-len("_config")]
                    if isinstance(value, dict):
                        updated_values[stripped_key] = update_config(cfg[stripped_key], value, overwritten_args.get(stripped_key, {}))
                    else:
                        if stripped_key not in overwritten_args:
                            cfg[stripped_key] = value
                            updated_values[stripped_key] = value
            return updated_values
        
         
        # Convert QtransChkptMetaData to dictionary
        print(f"Checkpoint metadata: {checkpoint_metadata}")
        checkpoint_dict = OmegaConf.to_container(OmegaConf.structured(checkpoint_metadata), resolve=True)
        
        # Filter keys that start with qtrans_ and end with _config
        filtered_checkpoint_dict = {k[len("qtrans_"):-len("_config")]: v for k, v in checkpoint_dict.items() if k.startswith("qtrans_") and k.endswith("_config")}
        
        print(f"Filtered checkpoint dict: {filtered_checkpoint_dict}")
        # Merge configurations: defaults -> checkpoint -> CLI
        # default_cfg = compose(config_name="config")
        merged_cfg = OmegaConf.merge(cfg, OmegaConf.create(filtered_checkpoint_dict))
        print(merged_cfg)
        print(OmegaConf.from_dotlist(cli_args))
        merged_cfg = OmegaConf.merge(merged_cfg, OmegaConf.from_dotlist(cli_args))
        cfg = merged_cfg
        ConfigSingleton().config = cfg

        # Find and log overwritten arguments from CLI
        overwritten_args = find_overwritten_args_from_cli(merged_cfg, cli_args)
        log.info(f"Overwritten arguments from CLI: {overwritten_args}")
        
        log.info(f"Final merged config: {OmegaConf.to_yaml(merged_cfg)}")
    
    wandb_init(cfg) 
    exit_code=0
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