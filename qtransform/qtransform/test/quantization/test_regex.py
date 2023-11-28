import unittest
from yaml import safe_load
from omegaconf import DictConfig
from typing import Dict, Union, List, Optional
import re
from dataclasses import dataclass
from logging import getLogger

log = getLogger(__name__)

@dataclass
class RegexConfig(): 
    valid: bool
    number_of_sublayers: int = 1 #if it is not valid
    regex_index: List[int] = None #items start at zero, if it is not a valid string just an empty list

    def __post_init__(self):
        if not isinstance(self.regex_index, list):
            self.regex_index = list()


#TODO: test suite for all layers instead of one large test case

REGEX_STRINGS_FILE = 'regex_strings.yaml'

class QuantizationregexTest(unittest.TestCase):
    """
        Testclass to verify that layers possibly containing regex strings are interpreted correctly.
        To do that, each layer within the layer string is split by LAYER_SEPERATOR_STRING and checked if 
        it can be parsed. A regular expression is interpreted with REGEX_SEARCH_PATTERN. It basically only
        detects a string as a regex if it has the syntax of a raw string (r'something'). 
        This test class does not check if layers can be found within a model with the expression. 
        Use QuantizationTest for that.
    """

    LAYER_SEPERATOR_STRING = r'(r\'[^\']+\'|[^\.]+)'
    REGEX_SEARCH_PATTERN = r'r\'([^\']+)\''

    def setUp(self):
        self.assertEqual(isinstance(self.regex_cfg, RegexConfig), True)
        self.assertEqual(isinstance(self.regex_layer_name, str), True)

    def tearDown(self):
        self.regex_cfg: RegexConfig = None
        self.regex_layer_name: str = None

    def test_regex_strings(self):
        #log.debug(f'Testing layer: {self.regex_layer_name}')
        self.assertEqual(isinstance(self.regex_layer_name, str), True)
        self.assertEqual(isinstance(self.regex_cfg, RegexConfig), True)
        #split string by its sublayers
        sublayers = re.findall(self.LAYER_SEPERATOR_STRING, self.regex_layer_name)
        #add all layers that are interpreted as a regex and compare it with the config
        regex_layers = list()
        #remember if string is valid
        valid = True
        for i, sublayer in enumerate(sublayers):
            #log.debug(f'going through layer: {sublayer}')
            is_regex = re.match(self.REGEX_SEARCH_PATTERN, sublayer)
            if is_regex:
                regex_layers.append(i)
            #special characters should be encapsulated in a regex term
            #if the string supposedly is valid, fail
            elif not sublayer.replace('_', '').isalnum():
                valid = False
        self.assertEqual(self.regex_cfg.valid, valid, self.regex_layer_name)    
        self.assertEqual(sorted(self.regex_cfg.regex_index), regex_layers, self.regex_layer_name)
        #TODO: when regex is not properly escaped, regex terms containing dot (.) are considered as multiple layers instead of one
        self.assertEqual(len(sublayers), self.regex_cfg.number_of_sublayers, self.regex_layer_name)

        def assertEqual(self, first, second, msg = self.regex_layer_name):
            """
                Simple overload in order to not pass the layer name onto every single assertion
            """
            super().assertEqual(first, second, f'\"{msg}\"')

    def runTest(self):
        self.test_regex_strings()

def suite(filename: str = REGEX_STRINGS_FILE) -> unittest.TestSuite:
    
    return unittest.TestSuite(collect_testcases(filename=filename))

def collect_testcases(filename: str = REGEX_STRINGS_FILE) -> List[QuantizationregexTest]:
        with open(filename, 'r') as file:
            test_cases = list()
            layer_cfg_dict: Dict[str, RegexConfig] = safe_load(file)
            #cleanup
            for name, cfg in layer_cfg_dict.items():
                if not isinstance(cfg, RegexConfig):
                    try:
                        layer_cfg_dict[name] = RegexConfig(**cfg)
                    except Exception as e:
                        log.error(f'Regex Config for {name} is invalid. Reason: {e}')
                        raise e
                test_case = QuantizationregexTest()
                test_case.regex_cfg = layer_cfg_dict[name]
                test_case.regex_layer_name = name
                test_cases.append(test_case)
            log.info(f'Config specified: {layer_cfg_dict}')
            return test_cases

for case in collect_testcases():
    case.run()

unittest.TextTestRunner().run(suite())

