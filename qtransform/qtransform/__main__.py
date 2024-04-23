import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
from pickle import load
import logging
import qtransform
from qtransform.utils import addLoggingLevel
from pprint import PrettyPrinter
from qtransform import ConfigSingleton
from qtransform.utils.callbacks import Callback, call_on_run_start, call_on_run_end
import brevitas
from importlib import import_module
from typing import Dict

addLoggingLevel("TRACE", logging.DEBUG - 5, "trace")
log = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path=qtransform.get_module_config_path(), config_name="config.yaml")
def cli_wrapper(cfg: DictConfig):
    """ 
    this function exsists so that one can call qtransform from cli with prepending "python -m ".
    note that additional configs can be loaded via --config-dir https://github.com/facebookresearch/hydra/issues/874
    """
    ConfigSingleton().config = cfg
    main()

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def module_wrapper(cfg: DictConfig):
    ConfigSingleton().config = cfg
    main()

def get_callbacks(callback_cfg: DictConfig) -> Dict[str, Callback]:
    #hydra allows for a much more versatile callback configuration (via config.yaml), 
    #but for our purposes this is enough
    callbacks = {}
    if callbacks is None:
        log.info(f'No callbacks supplied')
        return callbacks
    for callback_name, callback_class in callback_cfg.items():
        split = callback_class.split('.')
        module: str = '.'.join(split[:-1])
        callback_class = split[-1]
        module = import_module(module)
        try:
            #callbacks should be a class and have no parameters in constructor
            callbacks[callback_name] = getattr(module, callback_class, None)()
        except:
            log.warning(f'Module {module.__name__ + "." + callback_class} not found', exc_info=True)
    return callbacks

def main():
    cfg = ConfigSingleton().config
    logging.captureWarnings(True)
    root_log = logging.getLogger("root")
    log = logging.getLogger(f"{__package__}.{__name__}")   
    if "trace" in cfg and cfg.trace:
        root_log.setLevel(logging.TRACE)
        log.warning("TRACE ENABLED")
    elif "debug" in cfg and cfg.debug:
        root_log.setLevel(logging.DEBUG)
        log.debug("DEBUG ENABLED")
    import json
    if hasattr(log, "trace"): log.trace("launched with config: " + json.dumps(OmegaConf.to_container(cfg), indent=2))
    log.debug(f'LAUNCHED WITH CONFIG: {cfg}')
    if "command" not in cfg.run:
        log.error("No run command found in run config, run config was: " + str(cfg.run))
        raise KeyError
    #call callbacks
    callbacks = get_callbacks(cfg.callbacks)
    call_on_run_start(callbacks)
    #config could have been changed by callbacks at this point
    cfg = ConfigSingleton().config
    #make sure that callbacks are still called after exceptions
    #this is an issue with our implementation of callbacks currently
    try:
        match cfg.run.command:
            case "train":          
                from qtransform.run import train
                train.run(cfg)
            case "bench":
                from qtransform.run import bench
                return  bench.run(cfg)
            case "infer":
                from qtransform.run import infer
                return  infer.run(cfg)
            case "export":
                from qtransform.run import export
                return  export.run(cfg)
            case "test":
                from qtransform.run import test
                return test.run(cfg)
            case _:
                log.error(f'Command "{cfg.run.command}" not recognized')
    except:
        log.critical("Script execution failed. Reason: ", exc_info=True)
    call_on_run_end(callbacks)

if __name__ == "__main__":
    module_wrapper()