import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from pickle import load
import logging
import qtransform
from qtransform.utils import addLoggingHandler, addLoggingLevel
from pprint import PrettyPrinter
from qtransform import ConfigSingleton
from qtransform.utils.callbacks import Callbacks
from qtransform.utils.helper import get_output_dir, write_to_pipe
import brevitas
from importlib import import_module
from typing import Dict
import sys
import wandb
from qtransform.utils import ID
from qtransform.wandb import wandb_init

addLoggingLevel("TRACE", logging.DEBUG - 5, "trace")
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=qtransform.get_module_config_path(), config_name="config.yaml")
def cli_wrapper(cfg: DictConfig):
    """ 
    this function exsists so that one can call qtransform from cli with prepending "python -m ".
    note that additional configs can be loaded via --config-dir https://github.com/facebookresearch/hydra/issues/874
    """
    ConfigSingleton().config = cfg
    addLoggingHandler(os.path.join(get_output_dir(), ID + ".log"))
    main()

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def module_wrapper(cfg: DictConfig):
    ConfigSingleton().config = cfg
    addLoggingHandler(os.path.join(get_output_dir(), ID + ".log"))
    main()

def main():
    # # check if execution is inside slurm
    # if 'SLURM_JOB_ID' not in os.environ:
    #     print("This script is not running inside a SLURM job.")
    # else:
    #     print(f"This script is running inside a SLURM job with SLURM_JOB_ID={os.environ['SLURM_JOB_ID']}")
        
    cfg = ConfigSingleton().config
    #remember which yaml files for each config group were chosen
    #useful to distinguish model outputs
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
    if cfg.wandb.enabled:
        wandb_init(cfg)    
    import json
    if hasattr(log, "trace"): log.trace("launched with config: " + json.dumps(OmegaConf.to_container(cfg), indent=2))
    log.info(f"Launch command: {sys.argv}")
    log.debug(f'LAUNCHED WITH CONFIG: {cfg}')
        
    if "command" not in cfg.run:
        log.error("No run command found in run config, run config was: " + str(cfg.run))
        raise KeyError
    #call callbacks
    callbacks = Callbacks(cfg.callbacks)
    callbacks.call_on_run_start(cfg)
    #config could have been changed by callbacks at this point
    cfg = ConfigSingleton().config
    #make sure that callbacks are still called after exceptions
    #this is an issue with our implementation of callbacks currently
    exit_code=0
    
    try:
        # TODO we cloud make this dynamic by importing by module name via a string 
        match cfg.run.command: 
            case "train":          
                from qtransform.run import train
                train.run(cfg)
            case "bench":
                from qtransform.run import bench
                bench.run(cfg)
            case "infer":
                from qtransform.run import infer
                infer.run(cfg)
            case "export":
                from qtransform.run import export
                export.run(cfg)
            case "test":
                from qtransform.run import test
                test.run(cfg)
            case "script":
                from qtransform.run import script
                script.run(cfg)
            case "analysis":
                from qtransform.run import analysis
                analysis.run(cfg)
            # TODO chaining options together inside
            #case "analysis":
            #    from qtransform.run import multi
            #    multi.run(cfg)
            case _:
                log.error(f'Command "{cfg.run.command}" not recognized')
    except Exception as e:
        exit_code = 1 # generic error
        log.critical("Script execution failed. Reason: ", exc_info=True)
        
        # OutOfMemoryError does not exsist in static python torch package...
        if e.__class__.__name__ == "OutOfMemoryError":
            exit_code = 2
    
    # make sure wandb is closed (gets checked internally wether wandb.run is activ) 
    wandb.finish(exit_code=exit_code)
    #unsure if config should be pickled if errors occured
    cfg = ConfigSingleton().config
    callbacks.call_on_run_end(cfg)

    if exit_code is not None and exit_code > 0:
        log.error(f"Exited with Status Code: {str(exit_code)}")
        exit(exit_code)
    log.info(f"Exited with Status Code: {str(exit_code)}")

if __name__ == "__main__":
    module_wrapper()