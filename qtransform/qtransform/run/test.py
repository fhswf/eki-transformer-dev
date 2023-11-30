
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
from abc import ABC

@dataclass 
class TestConfig():
    """
        Boilerplate for autocompletion
    """
    module: str
    filename: str

log = logging.getLogger(__name__)
def run(cfg: DictConfig):
    """ Runs unit tests for a certain scope (usually the modules within this package) specified in the yaml config. Each module represents a feature such as
        quantization, tokenization, datasets, models, optimization, export.
    """
    tests = cfg.run.tests
    log.info("================")
    log.info(f'TESTING IMPLEMENTATION')
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")
    runner = unittest.TextTestRunner()
    #test each package
    for test_cfg in tests:
        filenames = test_cfg.get('filenames')
        if filenames is None:
            log.error(f'Error for {test_cfg.module}: No file specified')
            raise KeyError
        elif test_cfg.get('module') is None:
            log.error(f'No module specified.')
            raise KeyError
        elif not isinstance(filenames, list):
            #log.warning(f'Field "filenames" for {test_cfg.module} is not a list. Trying to ')
            #filenames only contains one entry -> make it into a list
            filenames = list(filenames)
        for filename in filenames:
            try:
                module = importlib.import_module('qtransform.test.' + test_cfg.module)
                #every module has to contain a method called "suite" which returns a unittest.TestSuite instance with the test cases
                #the cases were configured from the filename, specified in the test.yaml config
                log.info(f'Currently testing: {test_cfg.module}')
                suite = getattr(module, "suite")
                result: unittest.result.TestResult = runner.run(suite(filename))
                log.info(f'Test results for {test_cfg.module}: Tests ran: {result.testsRun}. Errors: {len(result.errors)}. Failures: {len(result.failures)}')
            except Exception as e:
                log.error(f'Test suite for {test_cfg.module} failed. Reason: {e}.\nMaybe check the config file {filename} for typos or if it exists.')
                raise KeyError() #should an error be raised?
                