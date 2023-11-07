from copy import deepcopy
from brevitas import nn as qnn
from torch.nn import Module, ModuleDict
import logging
from qtransform.quantization import Quantizer, ActQuantArgs, BiasQuantArgs, WeightQuantArgs
from qtransform.classloader import get_data
import re
from typing import Dict
import inspect
from brevitas.export import export_qonnx

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
        #perform all property access operations with quantized model
        quantized_model: Module = model if inplace else deepcopy(model)
        #go through all submodules, then all layers within quant config 
        #for submodule_name, submodule_cfg in self.quant_cfg.modules.items():
        #name -> sublayerquantargs -> name -> sublayerquantargs...
        layers = self.quant_cfg.model_layers.layers
        #TODO: find out when that could happen
        if not layers:
            log.error(f'Quantization config for model {model} is not applicable')
            raise AttributeError
        
        #SublayerQuantArgs: name -> layers, name -> layers
        
        #transformer.wte
        #transformer.gelu
        #transformer.layer.attn.mha
        #transformer.layer.1.attn.mha
        #lin_1
        
        for submodule_name in layers:
            try:
                submodule: ModuleDict = quantized_model.get_submodule(submodule_name)
            except AttributeError:
                log.error(f'Passed model for quantization does not have submodule of name {submodule_name}')
                raise ValueError
            #go through each layer and perform quantization
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
                #replace current non-quantized layer with quantized layer
                submodule.add_module(layer_name, quantized_layer)
        return quantized_model
    def get_quantized_layer(self, layer: Module, layer_type: str, quantizers: Dict[str, type]):
        """
            Quantizes a layer as specified in the yaml config file for the corresponding model. 
        """
        #for now, layers in layer_type have to case match the pytorch layers e.g. ReLU instead of relu, Relu etc.
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
        for attribute_name in set(signature.parameters.keys()) - set(['self', 'dtype', 'device', 'inplace']):
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

    def train_qat(self, model: Module, function: any, args: list) -> Module:
        """
            Unlike pytorch, no special function has to be called in order to calibrate the qparams and train the model.
        """
        function(model, *args)
        raise NotImplementedError()

    def export_model(self, model: Module, filepath: str) -> None:
        #Idea: something along the lines of export_qonnx(model, export_path=filepath)
        raise NotImplementedError