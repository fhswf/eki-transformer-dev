import unittest
from qtransform.quantization import ModelQuantConfig, LayerQuantConfig, BaseQuant
from qtransform.quantization.brevitas_quant import BrevitasQuantizer
from qtransform.model.gpt import GPTConfig, GPT
from qtransform import device_singleton
import yaml
import torch
import brevitas.nn as qnn
from typing import Dict
from re import compile, search, findall, match
from dataclasses import fields
from enum import Enum
from logging import getLogger
from qtransform.utils.introspection import get_optional_type

#to make sure that quantization does not fail
device_singleton.device = 'cpu'
log = getLogger(__name__)

class TestQuantization(unittest.TestCase):
    """
        Testclass to verify correctness of quantization process for the GPT model.
    """
    def test_correct_cfg(self, model_cfg_path: str, model: torch.nn.Module) -> ModelQuantConfig:
        """
            Check if the model quant config from the yaml file is cleaned up properly
            into a ModelQuantConfig object
        """
        #quantized GELU not possible currently
        #TODO: make testing models more modular, specify model within config probably?
        #simulate dictconfig
        #'./gpt/test_gpt_2_small',
        with open(model_cfg_path,  'r') as yaml_file:
            yaml_quant_cfg: dict = yaml.safe_load(yaml_file)
            yaml_quant_cfg.update({"model":model,"device": 'cuda', "dtype": "Int8"})
            model_cfg = ModelQuantConfig(**yaml_quant_cfg)
        log.info(f'ModelQuantConfig object created for model config file: "{model_cfg_path}"')
        #check if args are correct except for layers
        for key, value in yaml_quant_cfg.items():
            if key == 'layers': 
                continue
            self.assertEqual(getattr(model_cfg, key), yaml_quant_cfg[key])
        log.info(f'Types attributes immediately within ModelQuantConfig are correct. Checking layers next.')
        #TODO: check regex
        """
        for layer_possibily_regex, layer_cfg in yaml_layers.items():
            #first of all, check if layer exists within model
            self.assertEqual(model.get_submodule(layer_possibily_regex), layer_cfg.layer)
            
        with self.assertRaises(AttributeError):
            model.get_submodule('transformer.wte')
        """
        #test cleanup of layer args
        model_cfg_layers = model_cfg.layers.keys()
        #first of all, check if all layers are applied within config
        #TODO: REGEX
        #self.assertEqual(set(model_cfg_layers) - set(yaml_quant_cfg["layers"].keys()), set())
        for layer_name, layer_cfg in model_cfg.layers.items():
            log.debug(f'Testing layer: {layer_name}')
            #yaml config can be a regex, need to check if it is regex
            yaml_cfg_layer = yaml_quant_cfg["layers"][layer_name]
            self.assertEqual(yaml_cfg_layer["layer_type"], layer_cfg.layer_type)
            #quantize does not have to be set. if not set, check if True
            yaml_quantize = yaml_cfg_layer.get("quantize")
            if yaml_quantize:
                self.assertEqual(yaml_quantize  ,layer_cfg.quantize)
            else:
                self.assertEqual(layer_cfg.quantize, True)
            self.assertNotEqual(match(layer_name, layer_cfg.name), None)
            #test quantization arguments for layer
            self.test_quantizer_cfg(layer_cfg=layer_cfg, yaml_cfg_layer_quantizer=yaml_cfg_layer.get("quantizers"))
            log.info(f'Tests for layer {layer_name} passed.')
        log.info(f'Tests for creation of ModelQuantConfig passed.')
        return model_cfg

    def test_quantizer_cfg(self, layer_cfg: LayerQuantConfig, yaml_cfg_layer_quantizer: dict):
        """
            Tests if quantizer arguments are applied correctly.
        """
        #check when no custom quantizers are applied
        if yaml_cfg_layer_quantizer is None:
            self.assertEqual(len(layer_cfg.quantizers.keys()), 0)
            return
        #check if quantizers are applied within config
        quantizers_layer = layer_cfg.quantizers.keys()
        self.assertEqual(set(quantizers_layer), set(yaml_cfg_layer_quantizer.keys()), set())
        #custom quantizers are completely arbitrary, so None type is always allowed
        for quant_name, quantizer_layer in layer_cfg.quantizers.items():
            log.debug(f'Testing quantizer: {quant_name}')
            #if quantizer is in dataclass, should come from yaml config
            yaml_quantizer_layer = yaml_cfg_layer_quantizer.get(quant_name)
            self.assertNotEqual(yaml_quantizer_layer, None)
            #if a quantizer was set, it is necessary to define the default quantizer
            self.assertNotEqual(yaml_quantizer_layer["default_quantizer"], None)
            self.assertEqual(yaml_quantizer_layer.get("default_quantizer"), quantizer_layer.default_quantizer)
            self.assertEqual(yaml_quantizer_layer.get("template"), quantizer_layer.template)
            #check if type is inferred correctly from name
            if yaml_quantizer_layer.get("type") is None:
                self.assertNotEqual(search(quantizer_layer.type, quant_name), None)
            else:
                self.assertEqual(yaml_quantizer_layer.get("type"), quantizer_layer.type)
            #check actual quantizer args
            yaml_quantargs = yaml_quantizer_layer.get("args")
            if yaml_quantargs is None:
                self.assertEqual(quantizer_layer.args, None)
                continue
            log.info(f'Tests for fields immediately within BaseQuant for quantizer {quant_name} passed.')
            for field in (x for x in fields(quantizer_layer.args) if x.name != 'zero_point_impl'):
                log.debug(f'Testing field: {field.name}')
                yaml_quantizer_arg = yaml_quantargs.get(field.name)
                quantizer_arg = getattr(quantizer_layer.args, field.name)
                #quantizer args are optional
                if yaml_quantizer_arg is None:
                    self.assertEqual(quantizer_arg, None)
                    continue
                #enum value is the same as string literal e.g. QuantType["BINARY"] == "BINARY" is True
                optional_type = get_optional_type(field.type)
                self.assertEqual(optional_type(yaml_quantargs.get(field.name)), quantizer_arg)
                #check if type is correct, important for brevitas' enum driven api
                log.critical(f'{type(quantizer_arg)}, {field.type}')
                #self.assertEqual(isinstance(quantizer_arg, field.type), True)
            log.debug(f'Testing field zero_point_impl')
            log.info(f'Tests for QuantArgs passed.')

    def test_correct_quantization(self, model_cfg_path: str, model: torch.nn.Module):
        """
            Check if the classes of layers are replaced with the correct quantized classes
        """
        model_cfg = self.test_correct_cfg(model_cfg_path, model=model)
        #for auto completion
        model_cfg: ModelQuantConfig = model_cfg
        model: torch.nn.Module = model
        for layer_cfg in model_cfg.layers.values():
            layer_to_be_quantized = model.get_submodule(layer_cfg.name)
            quantized_layer = BrevitasQuantizer.get_quantized_layer(layer_to_be_quantized, layer_cfg.layer_type, layer_cfg.get_custom_quantizers(), layer_cfg.name) 
            #check if quantized layer class is within qnn
            #depends on whether class (Linear) is not within brevitas.nn subpackage, but quantized class (QuantLinear) is
            self.assertNotEqual(getattr(qnn, quantized_layer.__class__.__name__, None), None)
            #model config contains unquantized layers by default, as long as it is not overwritten by using BrevitasQuantizer.get_quantized_module
            self.assertNotEqual(layer_cfg.layer, quantized_layer)
            #TODO: check if custom args are applied
            #problem: names are different than from config... (is_signed instead of signed, zero_point_impl attribute not even within config etc.)

    def test_everything(self, model_cfg_path: str, model: torch.nn.Module):
        """
            Runs all test cases.
        """
        self.test_correct_quantization(model_cfg_path, model)

    def test_small_gpt(self):
        """
            Runs all test cases for a small gpt2 Model for the test_gpt_2_small.yaml file. 
            The config for that is: GPTConfig(10, 10, 3, 2, 4, 0, False, False, 'ReLU').
        """
        model_cfg_path ='/'.join(__file__.split('/')[:-1]) + '/gpt/test_gpt_2_small.yaml'
        gpt_cfg_small = GPTConfig(10, 10, 3, 2, 4, 0, False, False, 'ReLU')
        gpt_2_small = GPT(gpt_cfg_small)
        self.test_everything(model_cfg_path=model_cfg_path, model=gpt_2_small)