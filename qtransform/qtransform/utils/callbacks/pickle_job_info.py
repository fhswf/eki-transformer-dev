from . import Callback
from omegaconf import DictConfig, open_dict
from logging import getLogger
from hydra.core.hydra_config import HydraConfig
import hydra
from typing import Any, Dict, Union
import os
import pickle
from pprint import PrettyPrinter
from pathlib import Path
from logging import getLogger
from qtransform.utils.helper import write_to_pipe

#slightly modified version of: hydra.experimental.callbacks
class PickleJobInfoCallback(Callback):
    """Pickle the job config/return-value in ${output_dir}/{config,job_return}.pickle"""

    output_dir: Path

    def __init__(self, to_pipe: str = None) -> None:
        self.log = getLogger(f"{__name__}.{self.__class__.__name__}")
        self.to_pipe = to_pipe

    def on_run_start(self, config: DictConfig, **kwargs: Any) -> None:
        self.log.info(f'Saving hydra config at the end of run script')

    def on_run_end(self, config: DictConfig, **kwargs: Any) -> None:
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
        #writing to pipe blocks process, meaning that this callback should be put last if to_pipe is set
        if self.to_pipe is not None:
            write_to_pipe(self.to_pipe, str(self.output_dir / filename))

    def _save_pickle(self, obj: Any, filename: str, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        assert output_dir is not None
        with open(str(output_dir / filename), "wb") as file:
            pickle.dump(obj, file, protocol=4)