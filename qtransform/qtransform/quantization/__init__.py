#dtypes = [torch.qint8, torch.quint8, torch.quint32]
from abc import ABC, abstractclassmethod
import logging
import sys
from torch.nn import Module
from omegaconf import DictConfig
from typing import Optional, Tuple, Union, Dict
from dataclasses import dataclass
from qtransform.classloader import get_data
from enum import Enum
from brevitas.inject.enum import QuantType, FloatToIntImplType, ScalingImplType, BitWidthImplType, RestrictValueType, StatsOp
from brevitas.jit import ScriptModule
from brevitas.core.zero_point import __all__


#TODO: add bias quantization config

"""
class QuantConfig(Enum):
    ACT: str = "act"
    WEIGHT: str = "weight"
    BIAS: str = "bias"
"""

@dataclass 
class QuantArgs:
    pass

@dataclass
class WeightQuantArgs(QuantArgs):
    quant_type : QuantType #Integer, binary, ternary, fixed point integer
    bit_width_impl_type : BitWidthImplType #is the bit width backpropagated and optimised
    float_to_int_impl_type : FloatToIntImplType #how should the quantized values be clipped to fit into the quantized datatype
    narrow_range : bool #clip max value of data type (e.g. for 8 bits: -128:127 instead of -127:127)
    signed : bool #can quantized values take on negative values
    zero_point_impl : ScriptModule #how is zero point infered

    scaling_impl_type : ScalingImplType #how is the scale calculated, for now: statistics

    #attributes only applied when scaling_impl_type is statistics
    scaling_stats_op : StatsOp #max value, minmax etc.
    scaling_min_val : float #minimum value that the scale is going to have during calibration

    scaling_per_output_channel : bool #per tensor or per channel quantization
    restrict_scaling_type : RestrictValueType #restrict range of values that scale qparam can have
    bit_width : int #bit width of quantized values

class BiasQuantArgs(QuantArgs):
    pass

@dataclass
class ActQuantArgs(QuantArgs):
    """
        WIP
    """
    max_value: Optional[float]
    min_value: Optional[float]

@dataclass
class LayerQuantArgs:
    quantize: bool
    kind: Optional[str]
    weight: Optional[WeightQuantArgs]
    bias: Optional[BiasQuantArgs]
    act: Optional[ActQuantArgs] #bias/ weight and act cancel each other out

#TODO: find out if Dict works with hydra
@dataclass
class SubModuleQuantArgs:
    name: str
    layers: Dict[str, LayerQuantArgs]

@dataclass
class ModelQuantArgs:
    name: str
    modules: Dict[str, SubModuleQuantArgs]

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

    def __init__(self, quant_cfg: ModelQuantArgs):
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
    log.warning(_quant_cfg)
    log.debug(f'Quantizing with parameters: {_quant_cfg}')
    quant_cfg = ModelQuantArgs(**_quant_cfg.model)
    log.debug(quant_cfg)
    sys.exit(100)
    #get_classes necessary, otherwise a circular import error will occur
    quantizer: Quantizer = get_data(log, package_self, quant_cfg.type, Quantizer, quant_cfg)
    return quantizer