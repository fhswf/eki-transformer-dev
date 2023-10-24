from brevitas import nn as qnn
from brevitas.quant import scaled_int as scaled_int
import brevitas
from torch.nn import Module
import logging
from qtransform.quantization import Quantizer
from qtransform.classloader import get_data
from typing import Tuple
from dataclasses import dataclass
from omegaconf import DictConfig
import re

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
                raise ValueError
            for layer_name, layer_cfg in submodule_cfg.layers.items():
                if not layer_cfg.quantize:
                    continue
                try:
                    layer: Module = getattr(submodule, layer_name)
                except AttributeError:
                    log.error(f'Submodule {submodule_name} does not have layer of name {layer_name}')
                    raise ValueError
                #actually quantize the layer
                submodule[layer_name] = self.get_quantized_layer(layer, cfg_act=layer_cfg.act, cfg_weight=layer_cfg.weight, cfg_bias=layer_cfg.bias)
        raise NotImplementedError()

    def get_quantized_layer(self, layer: Module, cfg_act, cfg_bias, cfg_weight) -> Module:
        """
            Gets the quantized equivalent of a layer tuned with optional quantization configuration.
        """
        layer_class = layer.__class__
        layer_class_name = re.split(r'\'', re.split(r'\.', str(layer_class))[-1])[0]
        #class_name: QuantLinear, QuantReLU... 
        quantized_layer = get_data(log=log, package_name=qnn, class_name='Quant' + layer_class_name, parent_class=layer_class)
        cfg_dict = dict()
        cfgs = [cfg_act, cfg_bias, cfg_weight]
        for cfg in cfgs:
            if cfg == None:
                continue
            for attr in dir(cfg):
                cfg_dict[attr] = cfg[attr]
        return quantized_layer(**cfg_dict)

    def train_qat(self, model: Module, function: any, args: list) -> Module:
        """
            Unlike pytorch, no special function has to be called in order to calibrate the qparams and train the model.
        """
        function(model, *args)
        raise NotImplementedError()

    def export_model(self, model: Module, filepath: str) -> None:
        raise NotImplementedError()