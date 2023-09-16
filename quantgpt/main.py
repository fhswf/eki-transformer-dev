import os
import hydra
from omegaconf import DictConfig
import logging


@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def main(cfg: DictConfig):
    log = logging.getLogger(__name__)
    if cfg.debug:
        root_log = logging.getLogger("root")
        root_log.setLevel(logging.DEBUG)
        log.debug("DEBUG ENABLED")
    log.info("launched with config: " + str(cfg))
    log.warning("App is not ready to run")

    match cfg.c:
        case "train":
            match cfg.train:
                case "":
                    pass            
            pass
        case "eval":
            pass
        case "bench":
            pass 
        case "infer":
            pass
        case _:
            print('Command not recognized')

if __name__ == "__main__":
    main()