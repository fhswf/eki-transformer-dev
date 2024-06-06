from datetime import datetime
import logging
from typing import Any, Dict, Generator, Union, List
from dataclasses import dataclass
from omegaconf import DictConfig, open_dict

import random
import json
import glob
import pickle
from collections import Counter

log = logging.getLogger(__name__)

def run(cfg : DictConfig):
    """ Selecting a python file via hydra cli to run with hydra config """
    log.info("==============")
    log.info("Running Script")
    log.info("==============")
    
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")
    