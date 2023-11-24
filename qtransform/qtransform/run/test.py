
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
import qtransform
from dataclasses import dataclass
import unittest
from qtransform.classloader import get_data

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
    qtransform_packages = {x.name:x for x in pkgutil.iter_modules(qtransform.__path__)}

    runner=unittest.TextTestRunner()
    #test each package
    for test_package_name in scope:
        log.info(f'Currently testing: {test_package_name}')
        if test_package_name not in qtransform_packages.keys():
            log.error(f'Module {test_package_name} was not found')
            raise KeyError
        test_suite = unittest.TestSuite()
        #one module can have multiple test iterations, each ahving different configs
        count = 0
        test_package = importlib.import_module('qtransform.test.' + qtransform_packages[test_package_name].name)
        test_class = "Test" + test_package_name.capitalize()
        for test_args in test_package_name:
            #TODO: pass args
            test_case: unittest.TestCase = get_data(log, test_package ,test_class , unittest.TestCase, test_args)
            test_case.run()
            #test_suite.addTest(test_case(str(count)))
        #runner.run(test_suite)