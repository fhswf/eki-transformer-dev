#dtypes = [torch.qint8, torch.quint8, torch.quint32]
from abc import ABC, abstractclassmethod
import logging
import sys
from torch.nn import Module
from omegaconf import DictConfig
from typing import Optional, Dict, get_args
from dataclasses import dataclass, fields
from qtransform.classloader import get_data
from enum import Enum
from brevitas.inject.enum import QuantType, FloatToIntImplType, ScalingImplType, BitWidthImplType, RestrictValueType, StatsOp
from brevitas.jit import ScriptModule
from brevitas.core.zero_point import __all__


#TODO: add bias quantization config

class QuantConfig(Enum):
    ACT: str = "act"
    WEIGHT: str = "weight"
    BIAS: str = "bias"


@dataclass 
class QuantArgs:
    pass

@dataclass
class WeightQuantArgs(QuantArgs):
    """
        Everything is optional as not every single quantization parameter has to be set in brevitas
    """
    quant_type : Optional[QuantType] #Integer, binary, ternary, fixed point integer
    bit_width_impl_type : Optional[BitWidthImplType] #is the bit width backpropagated and optimised
    float_to_int_impl_type : Optional[FloatToIntImplType] #how should the quantized values be clipped to fit into the quantized datatype
    narrow_range : Optional[bool] #clip max value of data type (e.g. for 8 bits: -128:127 instead of -127:127)
    signed : Optional[bool] #can quantized values take on negative values
    zero_point_impl : Optional[ScriptModule] #how is zero point infered

    scaling_impl_type : Optional[ScalingImplType] #how is the scale calculated, for now: statistics

    #attributes only applied when scaling_impl_type is statistics
    scaling_stats_op : Optional[StatsOp] #max value, minmax etc.
    scaling_min_val : Optional[float] #minimum value that the scale is going to have during calibration

    scaling_per_output_channel : Optional[bool] #per tensor or per channel quantization
    restrict_scaling_type : Optional[RestrictValueType] #restrict range of values that scale qparam can have
    bit_width : Optional[int] #bit width of quantized values

class BiasQuantArgs(QuantArgs):
    """
        WIP
    """
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
    from_template: Optional[bool] = None
    template: Optional[str] = None
    kind: Optional[str] = None
    weight: Optional[WeightQuantArgs] = None
    bias: Optional[BiasQuantArgs] = None
    act: Optional[ActQuantArgs] = None #bias/ weight and act cancel each other out

@dataclass
class ModelQuantArgs:
    name: str
    modules: Dict[str, Dict[str, LayerQuantArgs]]

    def __post_init__(self):
        """
            Check if the types are correct in order to prevent future issues with Brevitas. To do so,
            it iterates through the entire dict representation of the yaml config file and creates instances the corresponding
            dataclasses if necessary. For example, if a module is not of type LayerQuantArgs, the method creates an instance of
            LayerQuantArgs with the parameters supplied in the current version of the object.
        """
        if not isinstance(self.modules, Dict):
            log.error(f'Model config has to contain a dictionary of quantized submodules, not type: {type(self.modules)}.')
            raise TypeError
        for module_name, module_cfg in self.modules.items():
            if not isinstance(module_cfg, Dict):
                log.error(f'Submodule \"{module_name}\" has to be a dictionary, not {type(module_cfg)}')
                raise TypeError
            for layer_name, layer_cfg in module_cfg.items():
                if not isinstance(layer_cfg, LayerQuantArgs):
                    #convert dict to dataclass
                    try:
                        layer_cfg.quantize = bool(layer_cfg.quantize)
                        self.modules[module_name][layer_name] = layer = LayerQuantArgs(**layer_cfg)
                    except:
                        log.error(f'Layer {layer_name} has to contain property \"quantize\" of type boolean')
                        raise TypeError
                    if not layer.quantize:
                        continue
                    #type cleanup for all defined properties
                    for field in (x for x in fields(layer) if x.name != 'quantize'):
                        attr = getattr(layer, field.name)
                        if attr == None:
                            continue
                        #TODO: get type of attribute and call constructor
                        if not isinstance(attr, field.type) and isinstance(field.type, QuantArgs):
                            #Error: Need to unwrap Optional[type] into type for it to work
                            for quant_field in fields(field.type):
                                pass
                        elif not isinstance(attr, field.type):
                            #get_args(field.type)[0](attr): call constructor of type with arguments from attr
                            #for correct typing in attribute
                            #sort of buggy when type is str instead of Optional[str]
                            setattr(self.modules[module_name][layer_name], field.name, get_args(field.type)[0](attr))
                    #log.error(f'Layer \"{layer_name}\" has to be of type LayerQuantArgs, not {type(layer_cfg)}')
                    #raise TypeError

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
    #TODO: typecheck for brevitas and add type (weight, bias, act) to qparams
    quant_cfg = ModelQuantArgs(**_quant_cfg.model)
    log.debug(quant_cfg)
    #get_classes necessary, otherwise a circular import error will occur
    quantizer: Quantizer = get_data(log, package_self, quant_cfg.type, Quantizer, quant_cfg)
    return quantizer