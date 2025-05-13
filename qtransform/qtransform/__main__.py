import os
import hydra
from hydra.core.hydra_config import HydraConfig
from hydra import compose, initialize
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
    
    # before we start wandb we need to read out the checkpoint if we continue training
    checkpoint_location = cfg.get('model')
    checkpoint_location = checkpoint_location.get('checkpoint')
    if checkpoint_location is not None:
        # find overwritten arguments
        log.info(f"checkpoint {checkpoint_location} was given, loading meta data to continue from")
        checkpoint = load_checkpoint(checkpoint_location)
        checkpoint_metadata = checkpoint.get('qtrans_metadata')
        if checkpoint_metadata is None:
            raise ValueError(f"checkpoint {checkpoint_location} does not contain qtransform metadata")
        if not isinstance(checkpoint_metadata, QtransChkptMetaData):
            checkpoint_metadata = QtransChkptMetaData.from_dict(checkpoint_metadata)

        log.info(f"Overwritten arguments from checkpoint: {checkpoint_metadata.qtrans_hydra_overrides}")

        def merge_hydra_cli_args(checkpoint_overrides, cli_overrides):
            # Liste von Keys, die du NICHT aus dem Checkpoint übernehmen willst
            blocked_keys = ["run.epochs", "run.export"]  # erweiterbar nach Bedarf

            # Hilfsfunktion zum Validieren
            def is_valid_override(s):
                return isinstance(s, str) and "=" in s and not any(s.startswith(b + "=") for b in blocked_keys)

            # Checkpoint-Overrides filtern und in Dict umwandeln
            checkpoint_overrides_dict = {}
            for item in checkpoint_overrides:
                if is_valid_override(item):
                    key, value = item.split("=", 1)
                    checkpoint_overrides_dict[key] = value
                else:
                    log.info(f"Ignoring checkpoint override: {item!r}")

            # CLI-Overrides verarbeiten (werden bevorzugt)
            cli_overrides_dict = {}
            for item in cli_overrides:
                if "=" in item:
                    key, value = item.split("=", 1)
                    cli_overrides_dict[key] = value
                else:
                    log.warning(f"Ignoring invalid CLI override: {item!r}")

            # CLI überschreibt Checkpoint
            merged_overrides = {**checkpoint_overrides_dict, **cli_overrides_dict}

            # Rückgabe als List[str]
            return [f"{key}={value}" for key, value in merged_overrides.items()]


        merged_overrides_array = merge_hydra_cli_args(checkpoint_metadata.qtrans_hydra_overrides, hydra_cli_args)
        #merged_overrides_array = hydra_cli_args
        
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
