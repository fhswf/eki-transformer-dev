from copy import deepcopy
from brevitas import nn as qnn
from torch import device
from torch.nn import Module, ModuleDict
import logging
from qtransform.quantization import Quantizer, ModelQuantConfig
from qtransform.classloader import get_data
import re
from typing import Dict
import inspect
from brevitas.export import export_qonnx
from pprint import PrettyPrinter
from qtransform import device_singleton

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
    def get_quantized_model(quant_cfg: ModelQuantConfig, inplace=False) -> Module:
        if quant_cfg is None:
            log.error(f'Quantization needs to have both config and model')
            raise KeyError
        model = quant_cfg.model
        #perform all property access operations with quantized model
        quantized_model: Module = model if inplace else deepcopy(model)
        #go through all submodules, then all layers within quant config 
        #for submodule_name, submodule_cfg in self.quant_cfg.modules.items():
        #name -> sublayerquantargs -> name -> sublayerquantargs...
        layers = quant_cfg.layers
        #TODO: find out when that could happen
        if not layers:
            log.error(f'Quantization config for model {model} is not applicable')
            raise AttributeError
        #naively iterate through each layer name specified in config
        #which means that containers holding layers could be retrieved multiple times without caching
        for layer_cfg in [x for x in layers.values() if x.quantize]:
            if hasattr(log,"trace"): 
                log.trace(f'Quantizing layer : {PrettyPrinter(indent=1).pformat(layer_cfg.name)}') 
            else: 
                log.debug(f'Quantizing layer {layer_cfg.name}')
            layer_to_be_quantized: Module = quantized_model
            sublayer_names = layer_cfg.get_layers()
            #let the user know which layers specifically are not found within model
            existing_sublayer_names = str()
            for sublayer_name in sublayer_names:
                try:
                    layer_to_be_quantized = layer_to_be_quantized.get_submodule(sublayer_name)
                    existing_sublayer_names += sublayer_name + '.'
                except AttributeError:
                    error = f'Check layer \"{existing_sublayer_names}\"' if len(existing_sublayer_names) > 0 else f'The top layer {sublayer_names[0]} does not exist'
                    log.error(f'Passed model for quantization does not have submodule of name \"{layer_cfg.name}\". {error}')
                    raise ValueError
            #sublayer should now contain the layer to be quantized
            quantizers = layer_cfg.get_custom_quantizers()
            if hasattr(log,"trace"): log.trace(f'Custom quantizers for layer {layer_cfg.name}: {quantizers}')
            quantized_layer: Module = BrevitasQuantizer.get_quantized_layer(layer=layer_to_be_quantized, layer_type=layer_cfg.layer_type, quantizers=quantizers, layer_name=layer_cfg.name)
            #replace current non-quantized layer with quantized layer
            quantized_model.get_submodule('.'.join(sublayer_names[:-1])).add_module(sublayer_names[-1], quantized_layer)
        #remember that model within config is quantized
        quant_cfg.quantized = True if inplace else False
        return quantized_model
    
    def get_quantized_layer(layer: Module, layer_type: str, quantizers: Dict[str, type], layer_name: str = None):
        """
            Quantizes a layer as specified in the yaml config file for the corresponding model. 
        """
        #first of all, check if layer is quantized already
        if hasattr(qnn, layer.__class__.__name__):
            log.warning(f'Layer \"{layer_name}\" is already quantized, yet it is enabled for quantization in the config. Skipping for now.')
            #if quantized already, simply return it
            #possible feature: overwrite qparams of layer
            return layer
        #for now, layers in layer_type have to case match the pytorch layers e.g. ReLU instead of relu, Relu etc.
        quant_class = 'Quant' + layer_type
        try:
            #get quantized layers of generic modules
            quantized_layer_class: type = get_data(log=log, package_name=qnn, class_name=quant_class, parent_class=object)
        except KeyError:
            #quantize custom layers
            error = f'The config for layer {layer_name} specifies that it is of type \"{layer_type}\", however no quantized layer has been found within \"{qnn.__package__}\".'
            error += f'\nMaybe check spelling? (E.g. ReLU has to be specified as ReLU and not relu, Relu, ReLu...)'
            log.error(error)
            raise ValueError
        log.debug(f'Quantized layer found for \"{layer_name}\": \"{quantized_layer_class}\"')
        #retrieve all set hyperparameters of unquantized layer
        #usually supplied in constructor
        #exceptions: dtype, device have to be retrieved from general config
        signature_unquantized_layer = inspect.signature(layer.__init__)
        hyperparameters = dict()
        for attribute_name in set(signature_unquantized_layer.parameters.keys()) - set(['self', 'dtype', 'device', 'inplace']):
            #some init parameters are not necessarily stored as attributes in layer
            #e.g. dtype, device, _weight, ...
            try:
                hyperparameters[attribute_name] = getattr(layer, attribute_name)
            except AttributeError:
                pass
        #bias is not included in all layers, but is a required argument for some
        try:
            hyperparameters["bias"] = True if hyperparameters["bias"] is not None else False
        except:
            pass
        #check what quantizers are going to actually be applied during instantiation
        signature_quantized_layer = inspect.signature(quantized_layer_class.__init__)
        for i in set(quantizers.keys()) - set(x for x in signature_quantized_layer.parameters.keys() if x.find('_quant') != -1):
            log.warning(f'Quantizer for type: {i} of layer {layer_name} is not going to be applied as an argument of that name is not found within the constructor of {quantized_layer_class}.')
        args = {**hyperparameters, **quantizers}
        log.debug(f'Quantizing layer \"{layer_name}\" with args: \"{args}\"')
        #create object of quantized layer, passing hyperparameters from current layer and (custom) quantizer classes
        try:
            quantized_layer = quantized_layer_class(**args)
        except Exception as e:
            log.error(f'Quantization for layer \"{layer_name}\" unsuccessful. Reason:\n{e}. Maybe the specified quantizer needs more config args to be set?')
            raise ValueError
            #TODO: find good path for error messages
        return quantized_layer.to(device=device_singleton.device)

    def train_qat(model: Module, function: any, args: list) -> Module:
        """
            Unlike pytorch, no special function has to be called in order to calibrate the qparams and train the model.
        """
        function(model, *args)
        return model

    def export_model(model: Module, filepath: str) -> None:
        #Idea: something along the lines of export_qonnx(model, export_path=filepath)
        raise NotImplementedError