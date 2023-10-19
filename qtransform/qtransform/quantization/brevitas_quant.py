from brevitas import nn as qnn
import brevitas
from torch.nn import Module
import logging
from qtransform.quantization import Quantizer
from typing import Tuple

log = logging.getLogger(__package__)

class BrevitasQuantizer(Quantizer):
    def get_quantized_model(model: Module) -> Module:
        pass