from copy import deepcopy
from brevitas import nn as qnn
from torch import device
from torch.nn import Module, ModuleDict, Identity
import logging
from qtransform.quantization import Quantizer, ModelQuantConfig, LayerQuantConfig
from qtransform.classloader import get_data
import re
from typing import Dict, Tuple, List, Union
import inspect
from pprint import PrettyPrinter
from qtransform import device_singleton
from dataclasses import replace
from qtransform.quantization.quant_bn import replace_bn, QuantBatchnorm1d
from brevitas.nn.utils import merge_bn

#brevitas allows tweaking the quantization hyperparameters for each layer with the parameter weight_quant
#idea: pass these configs from hydra conf to each layer and override default configs found in brevitas.nn.scale_int
#the default for linear layers is for example Int8WeightPerTensorFloat (8 bits, int, per tensor, etc.)

log = logging.getLogger(__package__)

QUANTIZED_CLASSES = {x[0]:x[1] for x in inspect.getmembers(qnn,lambda x: inspect.isclass(x) and issubclass(x, Module))}

class BrevitasQuantizer(Quantizer):
    """
        Quantizes a model based on a specified hydra configuration based on our fork of the brevitas framework (https://github.com/fhswf/brevitas), using
        the branch fhswf-dev. It does this by replacing a torch layer with the corresponding brevitas layer (usually found by prepending the word Quant to
        the layer name).
    """
    def get_quantized_model(quant_cfg: ModelQuantConfig, inplace=False, suppress_logs = False) -> Tuple[Module, Union[ModelQuantConfig, None]]:
        log.info(f'Quantizing model')
        if quant_cfg is None:
            log.error(f'Quantization needs to have both config and model')
            raise KeyError
        model = quant_cfg.model
        #perform all property access operations with quantized model
        if not isinstance(inplace, bool):
            log.warning(f'Using wrong argument type for inplace (type {type(inplace)}. Assuming inplace=False)') if not suppress_logs else ""
            inplace=False
        quantized_model: Module = model if inplace else deepcopy(model)
        #go through all submodules, then all layers within quant config
        #name -> sublayerquantargs -> name -> sublayerquantargs...
        layer_cfgs = quant_cfg.layers.values()
        if not layer_cfgs:
            log.error(f'Quantization config for model {model} is not applicable')
            raise AttributeError
        #naively iterate through each layer name specified in config
        #which means that containers holding layers could be retrieved multiple times without caching
        for layer_cfg in [x for x in layer_cfgs if x.quantize and not x.replace_later]:
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
            quantized_layer: Module = BrevitasQuantizer.get_quantized_layer(layer=layer_to_be_quantized, layer_cfg=layer_cfg, model=model)
            #replace current non-quantized layer with quantized layer
            quantized_model.get_submodule('.'.join(sublayer_names[:-1])).add_module(sublayer_names[-1], quantized_layer)
        #remember that model within config is quantized
        quant_cfg.quantized = True if inplace else False
        #make quantization of layers to be replaced later easy by creating a new ModelQuantConfig instance
        #this should be particularly useful for quantizing batchnorm during export
        replace_layers_later: Dict[str, LayerQuantConfig] = {layer_cfg.name:layer_cfg for layer_cfg in [replace(x) for x in layer_cfgs if x.replace_later]}
        for replace_layers_later_name in replace_layers_later.keys():
            replace_layers_later[replace_layers_later_name].replace_later = False
        #return None to avoid type checking empty ModelQuantConfig
        if len(replace_layers_later.keys()) == 0:
            replace_layers_later = None
        else:
            replace_layers_later: ModelQuantConfig = ModelQuantConfig(quant_cfg.cls, replace_layers_later, device_singleton.device, model)
            #logging "skipped" layers during export could be confusing
            log.info(f'Skipping quantization of layers {[x for x in replace_layers_later.layers.keys()]} as replace_later is set to True. ' \
                'Call get_quantized_layer() explicitly at a time when the layer should be quantized.') if not suppress_logs else ""
        #let the user know which layers were not quantized along their configs
        return (quantized_model, replace_layers_later)
    
    def get_quantized_layer(layer: Module, layer_cfg: LayerQuantConfig, model: Module = None) -> Module:
        """
            Quantizes a layer as specified in the yaml config file for the corresponding model.

            The model parameter is only necessary for merging a batchnorm layer into a previous layer as the layer to be merged into
            has to be retrieved from the overarching model. 
        """
        #first of all, check if layer is quantized already
        if hasattr(qnn, layer.__class__.__name__):
            log.warning(f'Layer \"{layer_name}\" is already quantized, yet it is enabled for quantization in the config. Skipping for now.')
            #if quantized already, simply return it
            #possible feature: overwrite qparams of layer
            return layer
        #
        layer_type: str = layer_cfg.layer_type
        quantizers = layer_cfg.get_custom_quantizers()
        if hasattr(log,"trace"): log.trace(f'Custom quantizers for layer {layer_cfg.name}: {quantizers}')
        layer_name: str = layer_cfg.name

        #use merge_bn for batchnorm, ignore brevitas classes
        if re.search(r'batchnorm', layer_type, re.IGNORECASE):
            merge_bn_name = layer_cfg.args.get('merge_bn', '')
            if isinstance(merge_bn_name, str) and len(merge_bn_name) > 0:
                log.debug(f'Merging batchnorm "{layer_name}"')
                #extract layer name from corresponding batchnorm transformer block
                #batchnorm could therefore be merged with any layer name within the same depth of the model
                bn_layer_name = layer_name.split('.')[:-1]
                bn_layer_name.append(merge_bn_name)
                bn_layer_name: str = '.'.join(bn_layer_name)
                try:
                    assert isinstance(model, Module), f'When merging batchnorm into another layer, the model cannot be of type {type(model)}.'
                    bn = model.get_submodule(bn_layer_name)
                    merge_bn(bn, layer)
                    #get_quantized_model expects the quantized layer to be returned, in this case the layer is unquantized
                    #and the params are merged into another layer
                    #TODO: maybe rewrite how get_quantized_model works
                    #return qnn.QuantIdentity() #previous layer does the job of batchnorm now
                    return bn
                except Exception as e:
                    log.error(f'Layer could not be merged with merge_bn set to "{merge_bn_name}, reason: "', exc_info=True)

            elif layer_cfg.args.get("replace_bn", False):
                log.debug(f'Replacing batchnorm "{layer_name}')
                #custom quantizers redundant when quantizing before export as default qparams are set to default
                new_bn = QuantBatchnorm1d(layer.num_features, **quantizers)
                bn: QuantBatchnorm1d = replace_bn(layer, new_bn, qat=True)
                return bn 
            else:
                log.warning(f'Quantization of batchnorm should be ')
                return layer

        #filter every class which contains name of layer to be quantized
        # -> MultiheadAttention: QuantMultiheadAttention, BatchNorm1d: BatchNorm1dQuantToScaleBias
        quantized_class_name = list(filter(lambda x: re.search(layer_type, x), QUANTIZED_CLASSES.keys()))
        if len(quantized_class_name) != 1:
            log.error(f'Found quantizer classes with layer_type "{layer_type}": {quantized_class_name}. Exactly one entry needs\
                to appear within this list, not {len(quantized_class_name)}')
            raise ValueError()
        quantized_layer_class = QUANTIZED_CLASSES[quantized_class_name[0]]
        log.debug(f'Quantized layer found for \"{layer_name}\": \"{quantized_layer_class}\"')
        #retrieve all set hyperparameters of unquantized layer
        #usually supplied in constructor
        #exceptions: dtype, device have to be retrieved from general config
        signature_unquantized_layer = inspect.signature(layer.__init__)
        signature_quantized_layer = inspect.signature(quantized_layer_class.__init__)
        log.debug(f'Constructor signature of layer {layer_name} (class {layer.__class__}): {signature_unquantized_layer}')
        log.debug(f'Constructor signature of quantized layer {quantized_class_name}: {signature_quantized_layer}')
        hyperparameters = dict()
        necessary_params = set(signature_unquantized_layer.parameters.keys()) & set(signature_quantized_layer.parameters.keys())
        for attribute_name in necessary_params - set(['self', 'dtype', 'device', 'inplace']):
            #some init parameters are not necessarily stored as attributes in layer
            #e.g. dtype, device, _weight, ...
            hyperparameter = getattr(layer, attribute_name, None)
            if hyperparameter is None:
                continue
            hyperparameters[attribute_name] = hyperparameter
        #bias is not included in all layers, but is a required argument for some
        if "bias" in necessary_params:
            hyperparameters["bias"] = True if hyperparameters.get("bias", None) is not None else False
        #check what quantizers are going to actually be applied during instantiation
        signature_quantized_layer = inspect.signature(quantized_layer_class.__init__)
        for i in set(quantizers.keys()) - set(x for x in signature_quantized_layer.parameters.keys() if x.find('_quant') != -1):
            log.warning(f'Quantizer for type: {i} of layer {layer_name} is not going to be applied as an argument of that name is not found within the constructor of {quantized_layer_class}.')
        args = {**hyperparameters, **quantizers}
        if layer_cfg.args is not None:
            args.update(**layer_cfg.args)
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
        return function(model, *args)