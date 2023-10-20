from brevitas import nn as qnn
import brevitas
from torch.nn import Module
import logging
from qtransform.quantization import Quantizer
from typing import Tuple

log = logging.getLogger(__package__)

class BrevitasQuantizer(Quantizer):
    """
        Quantizes a model based on a specified hydra configuration based on the brevitas framework (https://github.com/Xilinx/brevitas).
        As it stands, the dev branch of brevitas is used for quantization. As opposed to pytorch, brevitas offers native GPU
        quantization as well as allowing quantization on a specified number of bits. 
    """
    def get_quantized_model(self, model: Module) -> Module:
        raise NotImplementedError()

    def train_qat(self, model: Module, function: any, args: list) -> Module:
        raise NotImplementedError()

    def export_model(self, model: Module, filepath: str) -> None:
        raise NotImplementedError()