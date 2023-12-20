import os
import hydra
from omegaconf import DictConfig, OmegaConf
import logging
import qtransform
from qtransform.utils import addLoggingLevel
addLoggingLevel("TRACE", logging.DEBUG - 5, "trace")

@hydra.main(version_base=None, config_path=qtransform.get_module_config_path(), config_name="config.yaml")
def cli_wrapper(cfg: DictConfig):
    """ 
    this function exsists so that one can call qtransform from cli with prepending "python -m ".
    not that additional configs can be loaded via --config-dir https://github.com/facebookresearch/hydra/issues/874
    """
    main(cfg)

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def module_wrapper(cfg: DictConfig):
    main(cfg)

def main(cfg: DictConfig):
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
    
    if "command" not in cfg.run:
        log.error("No run command found in run config, run config was: " + str(cfg.run))
        raise KeyError
    match cfg.run.command:
        case "train":          
            from qtransform.run import train
            return  train.run(cfg)
        case "bench":
            from qtransform.run import bench
            return  bench.run(cfg)
        case "infer":
            from qtransform.run import infer
            return  infer.run(cfg)
        case "inferonnx":
            from qtransform.run import inferonnx
            return  inferonnx.run(cfg)
        case "export":
            from qtransform.run import export
            return  export.run(cfg)
        case "test":
            from qtransform.run import test
            return test.run(cfg)
        case _:
            log.error(f'Command "{cfg.run.command}" not recognized')

if __name__ == "__main__":
    module_wrapper()