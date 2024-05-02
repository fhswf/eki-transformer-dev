from . import Callback
from omegaconf import DictConfig
import os
import pickle
from pprint import PrettyPrinter
import hydra
from logging import getLogger
from typing import Any

class FromFileInfoCallback(Callback):
    "Updates the model.from_file field to suit the newly generated checkpoint/ onnx model"

    def __init__(self) -> None:
        self.log = getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        #######################
        from_previous_run = config.from_previous_run
        if from_previous_run is None:
            return
        log.info(f'Updating config with from_previous_run={from_previous_run}')
        if os.path.isfile(from_previous_run):
            log.warn(f'from_previous_run expects directory path, not filepath. Removing filename')
            output_dir, _ = os.path.split(from_previous_run)
        elif os.path.isabs(from_previous_run):
            output_dir = from_previous_run
        else:
            #remove current timestamp
            output_dir, _ = os.path.split(hydra.core.hydra_config.HydraConfig.get().runtime.output_dir)
            output_dir = os.path.join(output_dir, from_previous_run)
        config_path = os.path.join(output_dir, 'config.pickle')
        with open(config_path, 'rb') as input:
            new_config = pickle.load(input)
        assert isinstance(config, DictConfig), f'Pickle file from {from_previous_run} is not a DictConfig file'
            #del config["run"]
        log.debug(f'Loaded config from previous run: {PrettyPrinter(indent=1).pformat(config)}')
        #unsure in what way the HydraConfig() config could be used for our purposes. for now, only use cfg from hydra.main()
        current_cfg = ConfigSingleton().config
        with open_dict(new_config):
            #change run type and callbacks, keep everything else from previous run
            new_config.run = current_cfg.run
            new_config.callbacks = current_cfg.callbacks
        #instead of updating HydraConfig(), update own config Singleton
        ConfigSingleton().config = new_config

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        return None
