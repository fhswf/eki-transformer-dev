import os
import hydra
from omegaconf import DictConfig
import logging

@hydra.main(version_base=None, config_path="../conf", config_name="config.yaml")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    if cfg.debug:
        root_log = logging.getLogger("root")
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
        case "eval":
            from qtransform.run import test
            return  test.run(cfg)
        case "bench":
            pass 
        case "infer":
            pass
        case _:
            log.error(f'Command "{cfg.run.command}" not recognized')

if __name__ == "__main__":
    main()