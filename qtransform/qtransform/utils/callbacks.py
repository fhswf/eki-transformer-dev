from qtransform import ConfigSingleton
from omegaconf import DictConfig, open_dict
from logging import getLogger
from hydra.core.hydra_config import HydraConfig
import hydra
from typing import Any, Dict
import os
import pickle
from pprint import PrettyPrinter
from pathlib import Path

log = getLogger(__name__)

#hydra defines their own callbacks which are called and customized based on the config.yaml file,
#however the config they get cannot be changed
#this is not useful when changing the from_file field
class Callback:
    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in RUN mode before job/application code starts. `config` is composed with overrides.
        Some `hydra.runtime` configs are not populated yet.
        See hydra.core.utils.run_job for more info.
        """
        ...

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
        """
        Called in RUN mode after job/application code returns.
        """
        ...


def call_on_run_start(callbacks: Dict[str, Callback]):
    cfg = ConfigSingleton().config
    for callback_name, callback in callbacks.items():
        log.debug(f'Calling {callback_name}')
        callback.on_run_start(cfg)

def call_on_run_end(callbacks: Dict[str, Callback]) -> Dict[str, Any]:
    """
    Calls callbacks at the end of the run script and collects the return values of each of them.
    """
    cfg = ConfigSingleton().config
    args = {}
    for callback_name, callback in callbacks.items():
        log.debug(f'Calling {callback_name}')
        arg = callback.on_run_end(cfg)
        if arg is not None:
            args[callback_name] = arg
    return args

#overall, callbacks arent that useful for manipulating the state of the application, but they could be useful for type checking
#the problem is that callbacks are called with the config at the start of the run script and do not reflect changes made to it
#it is possible to define our own callback interface and use them within the main script
#it does the same thing but it retrieves config via HydraConfig inbetween each event, thereby allowing changes
#it is not as configurable though
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

#from: hydra.experimental.callbacks
class PickleJobInfoCallback(Callback):
    """Pickle the job config/return-value in ${output_dir}/{config,job_return}.pickle"""

    output_dir: Path

    def __init__(self) -> None:
        self.log = getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        log.info(f'Saving hydra config at the end of run script')

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> str:
        """
        Pickle the job's config in ${output_dir}/config.pickle.
        It is saved at the end in order to reflect dynamic changes in the config
        """
        self.output_dir = Path(HydraConfig.get().runtime.output_dir)
        filename = "config.pickle"
        #save runtime choices (run=train, dataset=something etc...)
        with open_dict(config):
            config["runtime"]["choices"] = HydraConfig().get().runtime.choices
        self._save_pickle(obj=config, filename=filename, output_dir=self.output_dir)
        self.log.info(f"Saving job configs in {self.output_dir / filename}")
        return os.path.join(output_dir, filename)

    def _save_pickle(self, obj: Any, filename: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir is not None
        with open(str(output_dir / filename), "wb") as file:
            pickle.dump(obj, file, protocol=4)

class ToPipeCallBack(Callback):
    """
    Reads and writes content from a named pipe.
    The content could be a pickled config file to replicate previous runs.
    """
    def __init__(self) -> None:
        self.log = getLogger(f"{__name__}.{self.__class__.__name__}")

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        log.info(f'Printing generated config pickle file into pipe at successful execution of run script')

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> str:
        pass