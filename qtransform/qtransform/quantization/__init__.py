#dtypes = [torch.qint8, torch.quint8, torch.quint32]
from abc import ABC, abstractclassmethod
import logging
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple
from dataclasses import dataclass
from qtransform.classloader import get_data

@dataclass
class QuantArgs:
    dtype: str
    observer: str
    scheme: str
    scope: str

@dataclass
class QuantConfig():
    quantize: bool
    type: str
    kind: str
    device: str
    args: QuantArgs


"""
    Supported torch.nn modules for quantization by Brevitas:
    quant_eltwise, quant_convtranspose, quant_max_pool, quant_accumulator, 
    quant_rnn, quant_linear, quant_activation, quant_avg_pool, quant_upsample, equalized_layer, utils, quant_mha, quant_embedding, 
    quant_dropout, quant_conv, quant_bn, quant_scale_bias, hadamard_classifier, __init__, quant_layer
"""

class Quantizer(ABC):
    """
        A generic wrapper to handle QAT differently depending on the chosen framework specified in the hydra config.
        Currently, torch and brevitas are supported with torch being limited to training only on cpu backends.
        As it stands right now, brevitas should be chosen for QAT related purposes.
    """

    def __init__(self, quant_cfg: QuantConfig):
        self.quant_cfg = quant_cfg
    
    @abstractclassmethod
    def get_quantized_model(self, model: Module) -> Module:
        """
            Prepares a model for QAT by applying qparams to the corresponding layers of the model specified in the
            quant_cfg. 
        """
        pass

    @abstractclassmethod
    def train_qat(self, model: Module, function: any, args: list) -> Module:
        """    
            Performs QAT on a model that has been prepared with get_quantized_model. During training,
            the qparams are calibrated in order to turn the weights into the quantized datatype. 

        """
        pass

    @abstractclassmethod
    def export_model(self, model: Module, filepath: str) -> None:
        pass

log = logging.getLogger(__name__)
import qtransform.quantization as package_self

def get_quantizer(_quant_cfg: DictConfig) -> Quantizer:
    log.debug(f'Quantizing with parameters: {_quant_cfg}')
    quant_cfg = QuantConfig(**_quant_cfg)
    #get_classes necessary, otherwise a circular import error will occur
    quantizer: Quantizer = get_data(log, package_self, quant_cfg.type, Quantizer, quant_cfg)
    return quantizer