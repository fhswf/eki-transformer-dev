import unittest
from yaml import safe_load
from omegaconf import DictConfig
from typing import Dict, Union, List, Optional
import re
from dataclasses import dataclass
from logging import getLogger
from qtransform.quantization.model_regex_filter import search_layers_from_strings, compile_pattern_from_layerstring, REGEX_SEARCH_PATTERN, LAYER_SEPERATOR_STRING

log = getLogger(__name__)


@dataclass
class ModelConfig():
    hits: int = 0
    layers: List[str] = None #initialized during cleanup, not necessary to be supplied in config for regex_strings
    
    def __post_init__(self):
        if not isinstance(self.layers, list):
            self.layers = list()

@dataclass
class RegexStringTest(): 
    valid: bool
    name: str
    number_of_sublayers: int = 1 #if it is not valid
    regex_index: List[int] = None #items start at zero, if it is not a valid string just an empty list
    models: Dict[str, ModelConfig] = None

    def __post_init__(self):
        if not isinstance(self.regex_index, list):
            self.regex_index = list()
        if self.models is not None and not isinstance(self.models, dict):
            self.models = dict(**self.models)
        elif self.models is None:
            self.models = dict()

from os.path import join
REGEX_STRINGS_FILE = join('/'.join(__file__.split('/')[:-1]), 'cfg_files', 'regex_strings.yaml')


class QuantizationregexTest(unittest.TestCase):
    """
        Testclass to verify that layers possibly containing regex strings are interpreted correctly.
        To do that, each layer within the layer string is split by LAYER_SEPERATOR_STRING and checked if 
        it can be parsed. A regular expression is interpreted with REGEX_SEARCH_PATTERN. It basically only
        detects a string as a regex if it has the syntax of a raw string (r'something'). 
        This test class does not check if layers can be found within a model with the expression. 
        Use QuantizationTest for that.
    """

    def setUp(self):
        self.assertEqual(isinstance(REGEX_SEARCH_PATTERN, str), True)
        self.assertEqual(isinstance(LAYER_SEPERATOR_STRING, str), True)
        self.assertEqual(isinstance(self.regex_cfg, RegexStringTest), True)
        self.assertEqual(isinstance(self.regex_layer_name, str), True)

    def tearDown(self):
        self.regex_cfg: RegexStringTest = None
        self.regex_layer_name: str = None

    def test_regex_strings(self):
        #log.debug(f'Testing layer: {self.regex_layer_name}')
        self.assertEqual(isinstance(self.regex_layer_name, str), True)
        self.assertEqual(isinstance(self.regex_cfg, RegexStringTest), True)
        try: 
            #check if syntax is okay
            #if syntax is not okay, every other test should not be checked
            search_result = compile_pattern_from_layerstring(self.regex_layer_name)
            self.assertEqual(sorted(self.regex_cfg.regex_index), search_result.regex_index)
            self.assertEqual(search_result.number_of_sublayers, self.regex_cfg.number_of_sublayers)
            #check if correct layers are found
            for model_name, model_cfg in self.regex_cfg.models.items():
                found_layers = search_layers_from_strings(self.regex_layer_name, model_cfg.layers)
                self.assertEqual(model_cfg.hits, len(found_layers), self.regex_layer_name + " : " + model_name)
        except:
            self.assertEqual(self.regex_cfg.valid, False)

        def assertEqual(self, first, second, msg = self.regex_layer_name):
            """
                Simple overload in order to not pass the layer name onto every single assertion
            """
            super().assertEqual(first, second, f'\"---- For regex string: {msg}\"')

    def runTest(self):
        self.test_regex_strings()

def suite(filename: str) -> unittest.TestSuite:
    """
        Creates multiple testcases from a config file and adds them to a test suite.
        The suite is returned and can be run with a unittest.runner.
    """
    return unittest.TestSuite(collect_testcases(filename=filename))

def collect_testcases(filename: str) -> List[QuantizationregexTest]:
        with open(filename, 'r') as file:
            test_cases = list()
            regex_cfg = safe_load(file)
            if regex_cfg.get("regex_strings") is None:
                log.error(f'No regex strings specified under field "regex_strings"')
                raise KeyError()
            regex_cfg_models = regex_cfg.get("models")
            if regex_cfg_models is None:
                log.warning(f'No layers specified to test the regular expressions on. Only the syntax of the regex strings are going to be tested.')
                regex_cfg_models = dict()
            #cleanup
            for name, cfg in regex_cfg.get("regex_strings").items():
                if not isinstance(cfg, RegexStringTest):
                    try:
                        cfg: RegexStringTest = RegexStringTest(name =  name, **cfg)
                        regex_cfg["regex_strings"][name] = cfg
                    except Exception as e:
                        log.error(f'Regex Config for {name} is invalid. Reason: {e}')
                        raise e
                #regex string should be tested on specified layers within config
                for model_name, model_args in cfg.models.items():
                    if model_name not in regex_cfg_models:
                        log.error(f'Regex String {name} references model {model_name} which does not exist within config.')
                        raise KeyError()
                    layers = regex_cfg["models"][model_name]
                    if not isinstance(layers, list):
                        log.error(f'Layers for model {model_name} to be tested is not a list')
                    cfg.models[model_name] = ModelConfig(layers = layers, **model_args)
                test_case = QuantizationregexTest()
                test_case.regex_cfg = regex_cfg["regex_strings"][name]
                test_case.regex_layer_name = name

                test_cases.append(test_case)
            log.debug(f'Config specified: {regex_cfg}')
            return test_cases
