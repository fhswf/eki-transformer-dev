import hydra
from omegaconf import DictConfig
import logging
from qtransform.utils import addLoggingLevel
addLoggingLevel("TRACE", logging.DEBUG - 5, "trace")

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    root_log = logging.getLogger("root")
    log = logging.getLogger(f"{__package__}.{__name__}")   
    if "trace" in cfg and cfg.trace:
        root_log.setLevel(logging.TRACE)
        log.debug("TRACE ENABLED")
    elif "debug" in cfg and cfg.debug:
        root_log.setLevel(logging.DEBUG)
        log.debug("DEBUG ENABLED")
    log.debug("launched with config: " + str(cfg))
    log.warning("App is not ready to run")
    
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
        case _:
            log.error(f'Command "{cfg.run.command}" not recognized')

if __name__ == "__main__":
    main()