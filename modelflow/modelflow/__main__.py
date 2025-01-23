import os
import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from hydra.utils import instantiate
import logging
import modelflow
from modelflow.command.common import OutputManager
from modelflow.utils import addLoggingHandler, addLoggingLevel
from modelflow.utils.helper import get_cwd
from modelflow.utils.id import ID
from modelflow import CFG
import sys
import json
# this has to happen before the logger get initialised
addLoggingLevel("TRACE", logging.DEBUG - 5, "trace")
log = logging.getLogger(__name__)

def launch_main(cfg: DictConfig):
    addLoggingHandler(os.path.join(get_cwd() , ID + ".log"))
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
    main(cfg)

@hydra.main(version_base=None, config_path=modelflow.get_module_config_path(), config_name="config.yaml")
def cli_wrapper(cfg: DictConfig):
    """ 
    this function exsists so that one can call qtransform from cli with prepending "python -m ".
    note that additional configs can be loaded via --config-dir https://github.com/facebookresearch/hydra/issues/874
    """
    launch_main(cfg)

@hydra.main(version_base=None, config_path="conf", config_name="config.yaml")
def module_wrapper(cfg: DictConfig):
    launch_main(cfg)
    

def main(cfg):
    if hasattr(log, "trace"): log.trace("launched with config: " + json.dumps(OmegaConf.to_container(cfg), indent=2))
    log.info(f"Launch command: {sys.argv}")
    log.info(f"modelflow ID: {ID}")
    log.info("Storing run config in global...")
    CFG(cfg)
    log.debug("config:")
    log.debug(CFG())
    
    # TODO check execution evironment
    
    exit_code=0
    try:
        # OutputManager is a global singelton accessed by OutputManager(), config gets upplied via global as well
        store = OutputManager()  # noqa: F841
        log.info("creating scheduler")
        scheduler = instantiate(cfg.scheduler)
        log.info("creating runc config")
        run_config = instantiate(cfg.run)
        log.info("starting scheduler")
        scheduler.run(run_config)
    except Exception as e:
        exit_code = 1 # generic error
        log.critical(f"Script execution failed. Reason: {e}", exc_info=True)

    if exit_code is not None and exit_code > 0:
        log.error(f"Exited with Status Code: {str(exit_code)}")
        exit(exit_code)
    log.info(f"Exited with Status Code: {str(exit_code)}")

    pass

if __name__ == "__main__":
    module_wrapper()