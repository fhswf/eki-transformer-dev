import logging
from typing import Any
from omegaconf import DictConfig, open_dict
import hydra
import os
from datetime import datetime

log = logging.getLogger(__name__)

def run(cfg : DictConfig, **kwargs):
    """ Chain commands together """
    log.info("=====================")
    log.info("Running command Chain")
    log.info("=====================")
    



