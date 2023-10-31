#dtypes = [torch.qint8, torch.quint8, torch.quint32]
from abc import ABC, abstractclassmethod
import logging
import sys
from torch.nn import Module
from omegaconf import DictConfig
from typing import List, Optional, Dict, Tuple, Union, get_args
from dataclasses import dataclass, fields
from qtransform.classloader import get_data
from enum import Enum
from brevitas.inject.enum import QuantType, FloatToIntImplType, ScalingImplType, BitWidthImplType, RestrictValueType, StatsOp
from brevitas.jit import ScriptModule
from brevitas.core.zero_point import __all__
import brevitas.quant.solver as brevitas_solver
from brevitas.quant.scaled_int import Uint8ActPerTensorFloat
from typing_inspect import get_origin

"""
    From brevitas.TMVCon: 

    What happens whenever we are passing a keyword argument (with prefix weight_, input_, output_, bias_ for a layer like QuantLinear, no prefix for an 
    activation layer like QuantReLU) is that we are overriding attributes of the underlying quantizer. To make things more compact and re-usable, 
    we can also simply define a new quantizer by inheriting from the one we are customizing.
    -> in order to pass qparams for layers whose quantizers are by default set to zero, a corresponding quantizer class needs to be passed to the layer
    -> otherwise, the qparams are ignored entirely
    -> it can be neglected for weight quantizers for layers and act quantizers for activations; however input, output and bias quantization will not be supported
    -> if we pass a class into the quantized layer with missing qparams (quant type ...), the program will crash
    -> using default quantizers is necessary
"""
class WeightQuantizer(Uint8ActPerTensorFloat):
    pass

class QuantConfig(Enum):
    ACT: Tuple[str, QuantArgs] = ("act", ActQuantArgs)
    WEIGHT: str = ("weight", WeightQuantArgs)
    BIAS: str = ("bias", BiasQuantArgs)


@dataclass 
class QuantArgs:
    """
        Class for configuration parameters that can be set for either weight, act and bias config.
    """
    pass

#TODO: check if args are none and use default args then
@dataclass
class WeightQuantArgs(QuantArgs):
    """
        Captures the weight parameters for layers.
        Everything is optional as not every single quantization parameter has to be set in brevitas and cannot be set in pytorch.
    """
    quant_type : Optional[QuantType] = None#Integer, binary, ternary, fixed point integer
    bit_width_impl_type : Optional[BitWidthImplType] = None# is the bit width backpropagated and optimised
    float_to_int_impl_type : Optional[FloatToIntImplType] = None#how should the quantized values be clipped to fit into the quantized datatype
    narrow_range : Optional[bool] = None #clip max value of data type (e.g. for 8 bits: -128:127 instead of -127:127)
    signed : Optional[bool] = None #can quantized values take on negative values
    #zero_point_impl : Optional[ScriptModule] = None #how is zero point infered

    scaling_impl_type : Optional[ScalingImplType] = None #how is the scale calculated, for now: statistics

    #attributes only applied when scaling_impl_type is statistics
    scaling_stats_op : Optional[StatsOp] = None #max value, minmax etc.
    scaling_min_val : Optional[float] = None #minimum value that the scale is going to have during calibration

    scaling_per_output_channel : Optional[bool] = None #per tensor or per channel quantization
    restrict_scaling_type : Optional[RestrictValueType] = None #restrict range of values that scale qparam can have
    bit_width : Optional[int] = None #bit width of quantized values

class BiasQuantArgs(QuantArgs):
    """
        WIP
    """
    #test: Optional[float] = None
    pass

@dataclass
class ActQuantArgs(QuantArgs):
    """
        WIP
    """
    max_value: Optional[float] = None
    min_value: Optional[float] = None

@dataclass
class LayerQuantArgs:
    quantize: bool
    template: Optional[Dict[str, str]] = None
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
        if not isinstance(self.modules, Union[Dict, DictConfig]):
            log.error(f'Model config has to contain a dictionary of quantized submodules, not type: {type(self.modules)}.')
            raise TypeError
        #TODO: clean up
        for module_name, module_cfg in self.modules.items():
            if not isinstance(module_cfg, Union[Dict, DictConfig]):
                log.error(f'Submodule \"{module_name}\" has to be a dictionary, not {type(module_cfg)}')
                raise TypeError
            for layer_name, layer_cfg in module_cfg.items():
                if not isinstance(layer_cfg, LayerQuantArgs):
                    try:
                        layer_cfg.quantize = bool(layer_cfg.quantize)
                    except:
                        log.error(f'Layer {layer_name} has to contain property \"quantize\" of type boolean')
                        raise TypeError
                    self.modules[module_name][layer_name] = layer = LayerQuantArgs(**layer_cfg)
                        #self.modules[module_name][layer_name] = layer = LayerQuantArgs(**layer_cfg)
                    if not layer.quantize:
                        continue
                    #type cleanup for all defined properties
                    for field in (x for x in fields(layer) if x.name != 'quantize'):
                        log.debug(f'Cleaning up field ---{field.name}---')
                        attr = getattr(layer, field.name)
                        if attr == None:
                            continue
                        log.debug(f'Value of ---{field.name}--- is: {attr}')
                        #unwrap Optional[List[str]] to List[str]
                        field_type = get_args(field.type)
                        field_type = field_type[0] if isinstance(field_type, Union[List, Tuple]) else field_type
                        #from List[str] get type list to call constructor. type str should still be str and not None
                        origin_type = get_origin(field_type)
                        origin_type = origin_type if origin_type != None else field_type
                        log.debug(f'Supposed type: {field.type}, actual type: {field_type}, unwrapped type: {origin_type}')
                        if not isinstance(attr, origin_type):
                            new_attr: dict = attr
                            if origin_type in QuantArgs.__subclasses__():
                                log.debug(f'Cleaning up Quant arguments for type: {layer.kind}')
                                fields_key_type = {field.name:field.type for field in fields(origin_type)}
                                difference = set(fields_key_type.keys()) - set(new_attr.keys())
                                given_keys = set(fields_key_type.keys()) & set(new_attr.keys())
                                #all properties that are set to None within config
                                for quant_name in difference:
                                    new_attr[quant_name] = None
                                #cleanup for all properties set within config
                                for quant_name in given_keys:
                                    quant_type = fields_key_type[quant_name]
                                    new_attr[quant_name] = get_args(quant_type)[0](new_attr[quant_name])
                                """for quant_name, quant_type in {field.name:field.type for field in fields(origin_type)}.items():
                                    
                                    try:
                                        value = new_attr.get(quant_name)
                                        new_attr[quant_name] = get_args(quant_type)[0](value)
                                    except:
                                        #key not given -> set to None
                                        new_attr[quant_name] = None   
                                    log.debug(f'Cleaning up property: {quant_name}, {quant_type} and value: {new_attr[quant_name]}')
                                    #cast raw string to enum value ("INT" to QuantType.INT)
                                    #all quant arguments are optional by default, no checking needed"""
                                log.warning(new_attr)
                            #get_args(field.type)[0](attr): call constructor of type with arguments from attr
                            #example: call list with parameters within the config 
                            setattr(self.modules[module_name][layer_name], field.name, origin_type(new_attr))
                    #log.error(f'Layer \"{layer_name}\" has to be of type LayerQuantArgs, not {type(layer_cfg)}')
                    #raise TypeError

def get_origin_type(type: any) -> any:
    pass

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
    log.debug(f'Quantizing with parameters: {_quant_cfg}')
    #TODO: typecheck for brevitas and add type (weight, bias, act) to qparams
    quant_cfg = ModelQuantArgs(**_quant_cfg.model)
    log.debug(quant_cfg)
    sys.exit(100)
    #get_classes necessary, otherwise a circular import error will occur
    quantizer: Quantizer = get_data(log, package_self, quant_cfg.type, Quantizer, quant_cfg)
    return quantizer