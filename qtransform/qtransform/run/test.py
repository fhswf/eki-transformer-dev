
import logging
from typing import Any, Dict
from omegaconf import DictConfig
from qtransform.utils.helper import load_checkpoint
from qtransform.classloader import get_data
from torch.onnx import export
from datetime import datetime
import torch
#from qtransform.quantization.testing import test_quant
import pkgutil
import importlib
import qtransform.test
from dataclasses import dataclass
import unittest
from qtransform.utils.introspection import _get_module
from qtransform.test.quantization.regex import suite


@dataclass 
class TestConfig():
    """
        Boilerplate for syntax highlighting
    """
    module: str
    filename: str

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
    qtransform_test_packages = {x.name:x for x in pkgutil.iter_modules(qtransform.test.__path__)}
    runner = unittest.TextTestRunner()
    #test each package
    for test_package_name, test_cfgs in scope.items():
        log.info(f'Currently testing: {test_package_name}')
        if test_package_name not in qtransform_test_packages.keys():
            log.error(f'Module {test_package_name} was not found')
            raise KeyError()
        for test_module in test_cfgs:
            try:
                module = importlib.import_module('qtransform.test.' + test_package_name + "." + test_module.module)
                #every module has to contain a method called "suite" which returns a unittest.TestSuite instance with the test cases
                #the cases were configured from the filename, specified in the test.yaml config
            except Exception as e:
                log.error(f'Test suite for {test_module} failed. Reason: {e}')
                raise KeyError()
            suite = getattr(module, "suite")
            #TODO: somehow print info stuff before actual tests
            runner.run(suite(test_module.filename))