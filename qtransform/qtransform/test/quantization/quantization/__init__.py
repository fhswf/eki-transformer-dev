import unittest
from qtransform.quantization import ModelQuantConfig, LayerQuantConfig, BaseQuant, QuantArgs
from qtransform.quantization.brevitas_quant import BrevitasQuantizer
from qtransform.model.gpt import GPTConfig, GPT
from qtransform import device_singleton
import yaml
from torch import device
from torch.nn import Module
import torch.nn.modules as torch_modules

import brevitas.nn as qnn
from re import compile, search, findall, match
from dataclasses import fields
from enum import Enum
from typing import Dict, Any, Union, List
from omegaconf import DictConfig
from logging import getLogger
from qtransform.utils.introspection import get_optional_type
from dataclasses import dataclass
from qtransform.quantization.model_regex_filter import compile_pattern_from_layerstring

#to make sure that quantization does not fail
log = getLogger(__name__)

@dataclass
class Testmodel():
    cls: str
    args: Dict[str, Any]
    calc_loss_in_model: bool = False #arbitrary as training / inference is not tested

@dataclass
class Testargs():
    """
        Dataclass for arguments passed from the hydra config in order to:
            - construct a model
            - load quantization config
        Since the fields of the dataclass reflect the hydra fields and the user should not be confused with
        names such as yaml_quant_cfg, it does not contain the state of the test case.
    """
    config_file: str
    model: Testmodel
    valid: bool = True
    dtype: str = 'Int8' #arbitrary
    device: str = 'cpu' #arbitrary


QUANTIZED_LAYERS = {qnn}

from qtransform.model import get_model
#for now, each test case to be used has the name Test<Module>
class QuantizationTest(unittest.TestCase):
    """
        Testclass to verify if the quantization of a model for a certain config file is applied correctly.
        In order to test specific models, a class should be created that derives from this and contains
        the arguments in the structure of Testargs (ARGS).
    """
    ARGS: Testargs = None #since setting up a custom constructor for test cases is a headache, use ARGS and setUp/ tearDown to store state

    def setUp(self):
        log.debug(f'Setup for quantizaton testing')
        if self.ARGS is None:
            log.error(f'No arguments for quantization test passed.')
            raise KeyError()
        if isinstance(self.ARGS, Union[Dict, DictConfig]):
            self.ARGS = Testargs(**self.ARGS)
        self.model = get_model(model_cfg=self.ARGS.model)
        device_singleton.device = device(self.ARGS.device)
        #TODO: distinguish between relative and absolute path
        with open(self.ARGS.config_file,  'r') as yaml_file:
            self.yaml_quant_cfg: dict = yaml.safe_load(yaml_file)
            log.debug(self.yaml_quant_cfg)
            #dtype is arbitrary
            self.yaml_quant_cfg.update({"model":self.model,"device": device_singleton.device, "dtype":self.ARGS.dtype})
        log.info(f'Testing with ModelQuantConfig from config file: "{self.ARGS.config_file}"')
    
    def tearDown(self):
        """
        
        """
        self.ARGS = None
        self.model= None
        self.yaml_quant_cfg = None
        
    def check_args(self):
        """
            Check if the objects that are used for the testing process are not None.
        """
        log.debug(f'Checking if args are not empty.')
        self.assertNotEqual(self.ARGS, None)
        self.assertNotEqual(self.model, None)
        self.assertNotEqual(self.yaml_quant_cfg, None)
        self.assertNotEqual(self.model_quant_cfg, None)


    def test_modelquant_cfg(self):
        """
            Tests entire configuration of ModelQuantConfig. This includes each dataclass field
            of ModelQuantConfig, LayerQuantConfig, BaseQuant and QuantArgs. The corresponding methods
            in this class are invoked to do so.
        """
        log.debug(f'Method test_modelquant_cfg')
        try:
            self.model_quant_cfg = ModelQuantConfig(**self.yaml_quant_cfg)    
            #check if args of ModelQuantConfig are applied correctly
            for key, value in self.yaml_quant_cfg.items():
                if key in ['layers']: 
                    continue
                self.assertEqual(getattr(self.model_quant_cfg, key), self.yaml_quant_cfg[key])
            log.info(f'Types attributes immediately within ModelQuantConfig are correct. Checking layers next.')
            #possible regex strings within config in order map the layerquantconfig names which have been
            #interpreted during parsing
            yaml_layers = self.yaml_quant_cfg["layers"].keys()
            #next, check each LayerQuantConfig entry 
            for layer_name, layer_quant_cfg in self.model_quant_cfg.layers.items():
                yaml_layer_quant_cfg = self.yaml_quant_cfg["layers"][layer_name]
                self.test_layerquant_cfg(layer_quant_cfg, yaml_layer_quant_cfg)
        except:
            #rudimentary check if the error is due to the same layer being configured multiple times
            #otherwise fail the test
            self.assertEqual(self.ARGS.valid, False)
        log.info(f'Tests passed.')

    def test_layerquant_cfg(self, layer_quant_cfg: LayerQuantConfig, yaml_layer_quant_cfg: Union[Dict, DictConfig]):
        """
            Tests if the configuration options for one layer is parsed correctly.
        """
        log.debug(f'Testing layer: {layer_quant_cfg.name}')
        #parse regex in order to find 
        compile_result = compile_pattern_from_layerstring(layer_quant_cfg.name)
        self.assertEqual(isinstance(yaml_layer_quant_cfg, Union[Dict, DictConfig]), True)
        self.assertEqual(isinstance(layer_quant_cfg, LayerQuantConfig), True)
        self.assertEqual(yaml_layer_quant_cfg["layer_type"], layer_quant_cfg.layer_type)

        #setting custom qparams for layer quantizers is optional
        yaml_quantize = yaml_layer_quant_cfg.get("quantize")
        if yaml_quantize:
            self.assertEqual(yaml_quantize  ,layer_quant_cfg.quantize)
        else:
            #if layer is specified in config and no quantize is given, assume layer should be quantized
            self.assertEqual(layer_quant_cfg.quantize, True)
        
        self.assertEqual(layer_quant_cfg.name, layer_quant_cfg.name)
        #check if regex is parsed correctly is done in regex module
        yaml_layer_quantizers = yaml_layer_quant_cfg.get("quantizers")
        if yaml_layer_quantizers is None:
            yaml_layer_quantizers = dict()
        self.assertEqual(isinstance(yaml_layer_quantizers, Union[Dict, DictConfig]), True)
        #check if all quantizers for the layer are stored within LayerQuantConfig
        #-> difference between keys is zero
        log.critical(layer_quant_cfg)
        log.critical(f'{set(layer_quant_cfg.quantizers.keys())} {set(yaml_layer_quantizers.keys())}')
        self.assertEqual(len(set(layer_quant_cfg.quantizers.keys()) - set(yaml_layer_quantizers.keys())), 0)
        
        #test BaseQuant quantizers within layer
        for layer_quantizer_name, layer_quantizer in layer_quant_cfg.quantizers.items():
            log.debug(f'Testing quantizer: {layer_quantizer_name}')
            yaml_layer_quantizer = yaml_layer_quantizers[layer_quantizer_name]
            self.test_quantbase_cfg(layer_name = layer_quantizer_name, layer_quantizer = layer_quantizer, yaml_layer_quantizer = yaml_layer_quantizer)
        log.info(f'Tests for layer {layer_quant_cfg.name} passed.')

    def test_quantbase_cfg(self, layer_name: str, layer_quantizer: BaseQuant, yaml_layer_quantizer: Union[Dict, DictConfig]):
        """
            Tests if the configuration options for a quantizer (weight, act, bias, input, output, etc.) are parsed correctly.
            To do so, the fields of a BaseQuant object are compared to a default dictionary representing the quantizer config
            of a hydra file. The name of a layer is needed as well in order to test if the type (act, weight etc.) is inferred correctly
            if no type has been explicitly set within the config.
        """
        log.debug(f'Testing BaseQuant')
        #field default_quantizer is the only mandatory argument when customizing a quantizer
        self.assertEqual(isinstance(layer_quantizer.default_quantizer, str), True)
        self.assertEqual(layer_quantizer.default_quantizer, yaml_layer_quantizer["default_quantizer"])
        self.assertEqual(layer_quantizer.template, yaml_layer_quantizer.get("template"))
        #check if type is inferred correctly from name
        if yaml_layer_quantizer.get("type") is None:
            self.assertNotEqual(search(layer_quantizer.type, layer_name), None)
        else:
            self.assertEqual(yaml_layer_quantizer.get("type"), layer_quantizer.type)
        #quantizer_module should not be set within yaml config
        yaml_quantizer_module = yaml_layer_quantizer.get("quantizer_module")
        if yaml_quantizer_module:
            self.assertEqual(layer_quantizer.quantizer_module, yaml_quantizer_module)
        #check if quantizer module even is within brevitas
        self.test_quantargs_cfg(layer_qparams=layer_quantizer.args, yaml_layer_qparams = yaml_layer_quantizer.get("args"))
        log.info(f'Tests for QuantBase passed')

    def test_quantargs_cfg(self, layer_qparams: QuantArgs, yaml_layer_qparams: Union[Dict, DictConfig]):
        """
            Tests if custom quantization args for a quantizer are parsed correctly, namely if strings are parsed
            to the corresponding brevitas enum counterparts.
        """
        #no custom qparams set -> should be None on both parsed config and original yaml config
        if layer_qparams is None:
            self.assertEqual(yaml_layer_qparams, None)
            log.info(f'Tests for QuantArgs passed.')
            return
        self.assertEqual(isinstance(layer_qparams, QuantArgs), True)
        self.assertEqual(isinstance(yaml_layer_qparams, Union[Dict, DictConfig]), True)
        log.debug(f'Testing qparams')
        if yaml_layer_qparams is None:
            self.assertEqual(layer_qparams, None)
        for field in (x for x in fields(QuantArgs) if x.name != 'zero_point_impl'):
            log.debug(f'Testing field: {field.name}')
            yaml_qparam = yaml_layer_qparams.get(field.name)
            layer_qparam = getattr(layer_qparams, field.name, None)
            #quantizer args are optional
            if yaml_qparam is None:
                self.assertEqual(layer_qparam, None)
                continue
            optional_type = get_optional_type(field.type)
            #numbers could be in scientific notation, so cast them to the according type
            self.assertEqual(optional_type(yaml_qparam), layer_qparam)
            #check if type is correct, important for brevitas' enum driven api
            self.assertEqual(optional_type, type(layer_qparam))
        log.debug(f'Testing field zero_point_impl')

        #########
        log.info(f'Tests for QuantArgs passed.')

    def test_layer_quantization(self):
        """
            Check if the classes of layers are replaced with the correct quantized classes
        """
        self.check_args()
        for layer_quant_cfg in self.model_quant_cfg.layers.values():
            layer_to_be_quantized = self.model.get_submodule(layer_quant_cfg.name)
            quantized_layer = BrevitasQuantizer.get_quantized_layer(layer_to_be_quantized, layer_quant_cfg.layer_type, layer_quant_cfg.get_custom_quantizers(), layer_quant_cfg.name) 
            #TODO: maybe put that field in LayerQuantConfig dataclass
            unquantized_layer_class = getattr(torch_modules, layer_quant_cfg.layer_type, None)
            self.assertNotEqual(unquantized_layer_class, None)
            #check if quantized layer class is within qnn
            #depends on whether class (Linear) is not within brevitas.nn subpackage, but quantized class (QuantLinear) is
            if isinstance(quantized_layer, Module):
                #if quantized module is derived from Module, it also derives from the non-quantized torch class
                self.assertEqual(isinstance(quantized_layer, unquantized_layer_class), True)
            else: #if it is not a Module, it has to be an act function
                self.assertNotEqual(getattr(quantized_layer, "act_quant", None), None)
                self.assertEqual(quantized_layer.act_quant.fused_activation_quant_proxy.activation_impl.__class__, unquantized_layer_class)
            #model config contains unquantized layers by default, as long as it is not overwritten by using BrevitasQuantizer.get_quantized_module
            self.assertNotEqual(layer_quant_cfg.layer, quantized_layer)
            #TODO: check if custom args are applied
            #problem: names are different than from config... (is_signed instead of signed, zero_point_impl attribute not even within config etc.)
        log.info(f'Quantization passed.')

    def runTest(self):
        """
            Tests the implementation of ModelQuantConfig dataclass and the quantization process implemented in brevitas_quantization.py.
            The corresponding methods (test_modelquant_cfg and test_layer_quantization) are used.
        """
        result = self.test_modelquant_cfg()
        if not result.wasSuccessful:
            self.fail()
        if self.model_quant_cfg:
            self.test_layer_quantization()
        else:
            log.warning(f'Skipping test for quantization since the configuration of ModelQuantConfig did not succeed.')

def suite(filename: str) -> unittest.TestSuite:
    """
        Creates multiple testcases from a config file and adds them to a test suite.
        The suite is returned and can be run with a unittest.runner.
    """
    return unittest.TestSuite(collect_testcases(filename=filename))

from yaml import safe_load
from omegaconf import OmegaConf #Dict items can be accessed as attributes with DictConfig class

ERROR_PREFIX = "Error with file: "

def collect_testcases(filename: str) -> List[QuantizationTest]:
    with open(filename, 'r') as file:
        test_file = safe_load(file)
    if test_file.get('test_cases') is None:
        log.error(f'{ERROR_PREFIX} "{filename}". No test cases specified for quantization.')
        raise KeyError
    elif not isinstance(test_file["test_cases"], list):
        log.error(f'{ERROR_PREFIX} "{filename}". Field "test_cases" is not a list.')
    test_cases = list()
    for test_case in test_file["test_cases"]:
        config_file = test_case.get("config_file")
        model = test_case.get("model")
        with open(model, 'r') as model_fio:
            #check if syntax works
            #model_cfg: Testmodel = Testmodel(**safe_load(model_fio))
            model_cfg = OmegaConf.create(safe_load(model_fio))
        test = QuantizationTest()
        test.ARGS = Testargs(config_file, model_cfg)
        test_cases.append(test)
    return test_cases