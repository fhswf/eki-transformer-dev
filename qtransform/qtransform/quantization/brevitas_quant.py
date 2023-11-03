from brevitas import nn as qnn
from torch.nn import Module
import logging
from qtransform.quantization import Quantizer, ActQuantArgs, BiasQuantArgs, WeightQuantArgs
from qtransform.classloader import get_data
import re
import torch.nn.functional as F
from typing import Dict

#brevitas allows tweaking the quantization hyperparameters for each layer with the parameter weight_quant
#idea: pass these configs from hydra conf to each layer and override default configs found in brevitas.nn.scale_int
#the default for linear layers is for example Int8WeightPerTensorFloat (8 bits, int, per tensor, etc.)

log = logging.getLogger(__package__)

class BrevitasQuantizer(Quantizer):
    """
        Quantizes a model based on a specified hydra configuration based on the brevitas framework (https://github.com/Xilinx/brevitas).
        As it stands, the dev branch of brevitas is used for quantization. As opposed to pytorch, brevitas offers native GPU
        quantization as well as allowing quantization on a specified number of bits among other hyperparameters specified in the .yaml files of this module.
    """
    def get_quantized_model(self, model: Module) -> Module:
        #go through all submodules, then all layers within quant config
        for submodule_name, submodule_cfg in self.quant_cfg.modules.items():
            try:
                submodule: Module = getattr(model, submodule_name)
            except AttributeError:
                log.error(f'Passed model for quantization does not have submodule of name {submodule_name}')
                #raise ValueError
            for layer_name, layer_cfg in submodule_cfg.items():
                if not layer_cfg.quantize:
                    continue
                try:
                    submodule = None
                    layer: Module = getattr(submodule, layer_name)
                except AttributeError:
                    log.error(f'Submodule {submodule_name} does not have layer of name {layer_name}')
                    #raise ValueError
                #actually quantize the layer
                quantizers = layer_cfg.get_custom_quantizers()
                #submodule[layer_name]
                #setattr()
                #setatt
                self.get_quantized_layer(layer_type=layer_cfg.layer_type, quantizers=quantizers)
                #todo: setattr = self.get_quantized_layer(layer, cfg_act=layer_cfg.act, cfg_weight=layer_cfg.weight, cfg_bias=layer_cfg.bias)
                #log.debug(f'Quantized layer {layer_name} as: {submodule[layer_name]}')
        raise NotImplementedError()

    def get_quantized_layer(self, layer_type: str, quantizers: Dict[str, type]):
        """
            Quantizes a layer as specified in the yaml config file for the corresponding model. 
        """
        log.debug(f'{quantizers}')
        #from linear to Linear as Classes in Brevitas are camel cased (QuantLinear, QuantReLU, QuantEmbedding etc.)
        #layer_type = layer_type.capitalize()
        quant_class = 'Quant' + layer_type
        try:
            #get quantized layers of generic modules
            quantized_layer_class = get_data(log=log, package_name=qnn, class_name=quant_class, parent_class=object)
        except KeyError:
            #quantize custom layers
            log.error(f'Module {quant_class} not found within {qnn.__package__}. Maybe check spelling? (ReLU has to be ReLU and not relu)')
            raise ValueError
        log.debug(f'Quantized layer found: {quantized_layer_class}')
        #TODO: retrieve current attributes from layer
        #**quantizers has properties <type>_quant, e.g. weight_quant, each having a custom implementation of a brevitas quantizer if needed
        PLACEHOLDER_FOR_LAYER_ARGS = None
        quantized_layer = quantized_layer_class(PLACEHOLDER_FOR_LAYER_ARGS, **quantizers)
        return quantized_layer


    @DeprecationWarning
    def deprecated_get_quantized_layer(self, layer: Module, cfg_act: ActQuantArgs, cfg_bias: BiasQuantArgs, cfg_weight : WeightQuantArgs) -> Module:
        """
            Gets the quantized equivalent of a layer tuned with optional quantization configuration.
        """
        layer_class = layer.__class__
        layer_class_name = re.split(r'\'', re.split(r'\.', str(layer_class))[-1])[0]
        quant_class = 'Quant' + layer_class_name
        #class_name: QuantLinear, QuantConv1d,...
        #TODO: implement using QuantIdentity
        try:
            #get quantized layers of generic torch modules
            quantized_layer_class = get_data(log=log, package_name=qnn, class_name=quant_class, parent_class=Module)
        except KeyError:
            #quantize custom layers
            log.debug(f'Module {quant_class} not found within {qnn.__package__}')
        cfg_dict = dict()
        cfgs = [cfg_act, cfg_bias, cfg_weight]
        for cfg in cfgs:
            if cfg == None:
                continue
            for attr in [x for x in dir(cfg) if not re.search(r'__.+__', x)]:
                #bit_width for weights turns to: weight_bit_width
                cfg_dict[attr] = cfg[attr]
        #TODO: pass other parameters from model into quantized version e.g. layer size etc.
        quantized_layer = quantized_layer_class(**cfg_dict)
        return quantized_layer

    def train_qat(self, model: Module, function: any, args: list) -> Module:
        """
            Unlike pytorch, no special function has to be called in order to calibrate the qparams and train the model.
        """
        function(model, *args)
        raise NotImplementedError()

    def export_model(self, model: Module, filepath: str) -> None:
        raise NotImplementedError()