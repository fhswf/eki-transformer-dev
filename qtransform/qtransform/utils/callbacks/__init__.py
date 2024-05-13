from qtransform import ConfigSingleton
from omegaconf import DictConfig, open_dict
from logging import getLogger
from hydra.core.hydra_config import HydraConfig
import hydra
from typing import Any, Dict, Union
import os
import pickle
from pprint import PrettyPrinter
from pathlib import Path
from dataclasses import dataclass, field
from importlib import import_module
from logging import getLogger

log = getLogger(__name__)


#hydra defines their own callbacks which are called and customized based on the config.yaml file,
#however the config they get cannot be changed
#this is not useful when changing the from_file field
class Callback:
    def __init__(self, **kwargs):
        pass

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

@dataclass
class CallbackConfig():
    """
    Additional dataclass for further arg support within config.
    """
    cls: str
    args: Dict[str, Any] = None

    def __post_init__(self):
        if not isinstance(self.args, Union[Dict, DictConfig]):
            self.args = {}

#decorating callbacks with a dataclass is slightly unnecessary as
#callbacks are called at the start and end of the script
@dataclass
class Callbacks():
    """
    
    """
    callbacks: Dict[str, Callback]
    #to support properties
    _callbacks: Dict[str, Callback] = field(init=False, repr = False)


    def __init__(self, callback_cfg: Union[Dict, DictConfig]):
        self.callbacks = callback_cfg

    @property
    def callbacks(self):
        return self._callbacks

    @callbacks.setter
    def callbacks(self, value: Union[Dict, DictConfig]):
        self._callbacks = self._get_callbacks(value)

    def _get_callbacks(self, callbacks_cfg: Union[Dict, DictConfig]) -> Dict[str, Callback]:
        #hydra allows for a much more versatile callback configuration (via config.yaml), 
        #but for our purposes this is enough
        callbacks = {}
        if not isinstance(callbacks_cfg, Union[Dict, DictConfig]):
            log.error(f'Invalid callbacks: {callbacks}')
            raise TypeError()
        for callback_name, callback_cfg in callbacks_cfg.items():
            log.info(callback_name)
            assert isinstance(callback_cfg, Union[Dict, DictConfig]), f'Config for callback: "{callback_class}" is not valid'
            callback_cfg: CallbackConfig = CallbackConfig(**callback_cfg)
            callback_class = callback_cfg.cls
            callback_args = callback_cfg.args
            #callback_cfg should have properties: class, args
            split = callback_class.split('.')
            module: str = '.'.join(split[:-1])
            callback_class = split[-1]
            module = import_module(module)
            try:
                #instantiate callbacks with specified args from hydra config
                callbacks[callback_name] = getattr(module, callback_class, None)(**callback_args)
                log.debug(f'Constructed class: {callback_class} with args: {callback_args}')
            except:
                log.warning(f'Module {module.__name__ + "." + callback_class} not found', exc_info=True)
        return callbacks

    def call_on_run_start(self, cfg) -> None:
        """
        Calls the on_run_start method of each callback within this instance.
        """
        for callback_name, callback in self.callbacks.items():
            log.debug(f'Calling on_run_start of: "{callback_name}"')
            callback.on_run_start(cfg)

    def call_on_run_end(self, cfg) -> None:
        """
        Calls callbacks at the end of the run script and collects the return values of each of them.
        """
        for callback_name, callback in self.callbacks.items():
            log.debug(f'Calling on_run_end of: "{callback_name}"')
            callback.on_run_end(cfg)
