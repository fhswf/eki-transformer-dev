from brevitas import nn as qnn
from torch.nn import Module, ModuleDict
import logging
from qtransform.quantization import Quantizer, ActQuantArgs, BiasQuantArgs, WeightQuantArgs
from qtransform.classloader import get_data
import re
import torch.nn.functional as F
from typing import Dict
import inspect


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
    def get_quantized_model(self, model: Module, inplace=False) -> Module:
        quantized_model: Module = Module() if inplace else model
        #go through all submodules, then all layers within quant config
        for submodule_name, submodule_cfg in self.quant_cfg.modules.items():
            try:
                submodule: ModuleDict = model.get_submodule(submodule_name)
            except AttributeError:
                log.error(f'Passed model for quantization does not have submodule of name {submodule_name}')
                raise ValueError
            #if a model with the name already exists, do nothing
            quantized_submodule = Module()
            quantized_model.add_module(submodule_name, quantized_submodule)
            for layer_name, layer_cfg in submodule_cfg.items():
                if not layer_cfg.quantize:
                    continue
                try:
                    layer: Module = submodule.get_submodule(layer_name)
                except AttributeError:
                    log.error(f'Submodule \"{submodule_name}\" does not have layer of name \"{layer_name}\"')
                    raise ValueError
                #actually quantize the layer
                quantizers = layer_cfg.get_custom_quantizers()
                quantized_layer: Module = self.get_quantized_layer(layer=layer, layer_type=layer_cfg.layer_type, quantizers=quantizers)
                quantized_submodule.add_module(layer_name, quantized_layer)

    def get_quantized_layer(self, layer: Module, layer_type: str, quantizers: Dict[str, type]):
        """
            Quantizes a layer as specified in the yaml config file for the corresponding model. 
        """
        log.debug(f'{quantizers}')
        #from linear to Linear as Classes in Brevitas are camel cased (QuantLinear, QuantReLU, QuantEmbedding etc.)
        #layer_type = layer_type.capitalize()
        quant_class = 'Quant' + layer_type
        try:
            #get quantized layers of generic modules
            quantized_layer_class: type = get_data(log=log, package_name=qnn, class_name=quant_class, parent_class=object)
        except KeyError:
            #quantize custom layers
            log.error(f'Module {quant_class} not found within {qnn.__package__}. Maybe check spelling? (E.g. ReLU has to be ReLU and not relu, Relu...)')
            raise ValueError
        log.debug(f'Quantized layer found: {quantized_layer_class}')
        #retrieve all set hyperparameters of unquantized layer
        #usually supplied in constructor
        #exceptions: dtype, device have to be retrieved from general config
        signature = inspect.signature(layer.__init__)
        hyperparameters = dict()
        for attribute_name in set(signature.parameters.keys()) - set(['self', 'dtype', 'device']): #- set(dtype):
            #attribute = signature.parameters[attribute_name]
            hyperparameters[attribute_name] = getattr(layer, attribute_name)
        #bias is not included in all layers, but is a required argument for some
        try:
            hyperparameters["bias"] = True if hyperparameters["bias"] is not None else False
        except:
            pass
        args = {**hyperparameters, **quantizers}
        log.debug(f'Quantizing layer with args: {args}')
        quantized_layer = quantized_layer_class(**args)
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