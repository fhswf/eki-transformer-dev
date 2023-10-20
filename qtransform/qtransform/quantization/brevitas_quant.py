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

        for module_name, quant_args in self.quant_cfg:
            pass
            #replace old submodule with quantized module
            new_module = ""
            model[module_name] = new_module

        raise NotImplementedError()

    def train_qat(self, model: Module, function: any, args: list) -> Module:
        """
            Unlike pytorch, no special function has to be called in order to calibrate the qparams and train the model.
        """
        function(model, *args)
        raise NotImplementedError()

    def export_model(self, model: Module, filepath: str) -> None:
        raise NotImplementedError()