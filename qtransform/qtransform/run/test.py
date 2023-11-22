
import logging
from typing import Any
from omegaconf import DictConfig
from qtransform.utils.helper import load_checkpoint
from torch.onnx import export
from datetime import datetime
import torch
#from qtransform.quantization.testing import test_quant
import pkgutil
import importlib
import qtransform
from abc import ABC, abstractmethod
import unittest
from qtransform.classloader import get_data
import qtransform.test as test_module
from qtransform.test.quantization import test_quant

class TestClass(unittest.TestSuite, ABC):
    """
        Boilerplate to make sure that test suites have a function that can be called for all scopes to be tested
    """
    @classmethod
    @abstractmethod
    def test_everything():
        pass



log = logging.getLogger(__name__)

def run(cfg: DictConfig):
    """ Runs unit tests for a certain scope (usually the modules within this package) specified in the yaml config. Each module represents a feature such as
        quantization, tokenization, datasets, models, optimization, export.
    """
    scope = cfg.run.scope
    log.info("================")
    log.info(f'TESTING IMPLEMENTATION')
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")
    qtransform_modules = {x.name:x for x in pkgutil.iter_modules(qtransform.__path__)}
    for test_suite in scope:
        log.info(f'Currently testing: {test_suite}')
        if test_suite not in qtransform_modules.keys():
            log.error(f'Module {test_suite} was not found')
            raise KeyError
        #get_data(log, )
        #log.critical(qtransform_modules[test_suite][1])
        #importlib.import_module(qtransform_modules[test_suite].module_finder)
        #log.critical(qtransform_modules)
        #module_scope = getattr(qtransform_package, test_suite, None)
        test_quant.TestQuantization().test_small_gpt()