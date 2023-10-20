#dtypes = [torch.qint8, torch.quint8, torch.quint32]
from abc import ABC, abstractclassmethod
import logging
from torch.nn import Module
from omegaconf import DictConfig
from typing import Tuple, Union
from dataclasses import dataclass
from qtransform.classloader import get_data
from enum import Enum
from brevitas.inject.enum import FloatToIntImplType


#TODO: refactor config in quantizers for new yaml config and create configs for each layer

class QuantScope(Enum):
    TENSOR = 1
    CHANNEL = 2

@dataclass
class GenericQuantArgs:
    signed: bool
    bit_width: 8 #if pytorch is used: bit_width restricted to 8 or 32 for int, 16 for float
    #https://pytorch.org/docs/stable/quantization-support.html#quantized-dtypes-and-quantization-schemes
    scheme: str #affine or symmetric
    #TODO: apparently for brevitas, minmax is applied with two properties instead of one
    #   -> scaling_impl_type = ScalingImplType.STATS to let the scale be based off of statisstics
    #   -> scaling_stats_op = StatsOp.MAX to make the clipping range be based off of minmax values
    scaling_impl_type: str #specify how the clipping range for the scale qparam is applied. minmax: take min and max value of scope
    #scope: either tensor or channel
    #tensor: qparams (scale and zero value) is applied to an entire layer; 
    #channel: qparams are applied for each weight within the layer
    scope: QuantScope
    ######PARAMS BELOW ONLY SUPPORTED BY BREVITAS#####
    quant_type: str #INT, BINARY, TERNARY, FP (floating point)
    bit_width_learnable: bool #if set to true: bit width is backpropagated and optimized (e.g. from 8 bits to 17 bits after calibration)
    float_to_int_impl_type: FloatToIntImplType #sets how quantized values are clipped, other options: CEIL, FLOOR, ROUND_TO_ZERO, DPU, LEARNED_ROUND
    zero_point_impl: str #zzpoint: zero qparam is always 0, other options: classes from brevitas.core.zero_point

@dataclass
class WeightQuantArgs(GenericQuantArgs):
    pass

@dataclass
class ActQuantArgs(GenericQuantArgs):
    max_value: Union[float, None]
    min_value: Union[float, None]

@DeprecationWarning
@dataclass
class QuantArgs:
    signed: bool
    bit_width: int
    #dtype: str
    observer: str
    scheme: str
    scope: str
    max_value: Union[int, None]
    min_value: Union[int, None]

@DeprecationWarning
@dataclass
class QuantConfig():
    quantize: bool
    type: str
    kind: str
    models: DictConfig
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