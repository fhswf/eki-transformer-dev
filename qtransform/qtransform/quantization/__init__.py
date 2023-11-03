#dtypes = [torch.qint8, torch.quint8, torch.quint32]
from abc import ABC, abstractclassmethod
import logging
import sys
from torch.nn import Module
from omegaconf import DictConfig
from typing import List, Optional, Dict, Tuple, Union, get_args, get_origin #get_args: get raw type of wrapper -> Optional[str] is (), Dict[str, str] = (str, str)...
from dataclasses import dataclass, fields
from qtransform.classloader import get_data
from qtransform.utils.introspection import get_optional_type
from enum import Enum
from brevitas.inject.enum import QuantType, FloatToIntImplType, ScalingImplType, BitWidthImplType, RestrictValueType, StatsOp
from brevitas.jit import ScriptModule
import brevitas.quant.solver as brevitas_solver

@dataclass 
class QuantArgs:
    """
        Class for configuration parameters that can be set for either weight, act, bias, input or output config.
        The class contains all parameters that can be configured for all kinds of quantizers, however not all are going to have values
        for them. Therefore, the datatype is by default set to None. when overriding the default quantizer, the attributes of value None
        should be ignored in order to avoid injection errors with brevitas.
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

    #if scaling_impl_type = ScalingImplType.PARAMETER_FROM_STATS and scaling_stats_op = StatsOp.PERCENTILE is used:
    high_percentile_q: Optional[float] = None #
    collect_stats_steps: Optional[int] = None #define the amount of steps needed to be taken to collect data for calculating scale qparam

    scaling_per_output_channel : Optional[bool] = None #per tensor or per channel quantization
    restrict_scaling_type : Optional[RestrictValueType] = None #restrict range of values that scale qparam can have
    bit_width : Optional[int] = None #bit width of quantized values

    def __post_init__(self):
        """
            Clean up the data types its attributes and turn the string values into its enum representations found in brevitas.
            For example, instead of quant_type being 'INT', it should be: <QuantType.INT: 'INT'>.
            This is necessary as brevitas uses injectors and relies on its enum driven API to dynamically configure quantization options.
        """
        for field in fields(self):
            #ignore None values
            current_value_for_field = getattr(self, field.name)
            if current_value_for_field is None:
                continue
            origin_type = get_optional_type(field.type)
            setattr(self, field.name, origin_type(current_value_for_field))
"""
    Below are Quantizer parameters which encapsulate qparams for the different kinds of quantizers.
"""
@dataclass
class WeightQuantArgs(QuantArgs):
    """
        Quantization parameters only existing within weights. 
    """
    pass

@dataclass
class BiasQuantArgs(QuantArgs):
    """
        Dataclass to encapsulate bias quantization parameters in order to override the default quantizer supplied in the config.
        The scale of biases can either be infered from both the weight and input scales or from a seperate scale for the bias.
    """
    requires_input_scale: Optional[bool] = None #bias quantization needs additional input quantization
    requires_input_bit_width: Optional[bool] = None

@dataclass
class ActQuantArgs(QuantArgs):
    """
        Dataclass to encapsulate act quantization parameters in order to override the default quantizer supplied in the config.
        Activation quantizers can clip output values into a certain range with max_val and min_val.
    """
    max_val: Optional[float] = None
    min_val: Optional[float] = None
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

#TODO: name things better

class QuantizationKind(Enum):
    ACT: str = "act"
    WEIGHT: str = "weight"
    BIAS: str = "bias"
    INPUT: str = "input"
    OUTPUT: str = "output"




#all quantized classes for easy type checking
#for now, only int is supported
from brevitas.quant.scaled_int import __all__ as all_int_quantizers
from brevitas.quant.scaled_int import __name__ as brevitas_scaled_int_module
from brevitas.quant.binary import __all__ as all_binary_quantizers
from brevitas.quant.binary import __name__ as brevitas_binary_module
from brevitas.quant.fixed_point import __all__ as all_fp_quantizers
from brevitas.quant.fixed_point import __name__ as brevitas_fixed_point_module
from brevitas.quant.ternary import __all__ as all_ternary_quantizers
from brevitas.quant.ternary import __name__ as brevitas_ternary_module

SUPPORTED_QUANTIZERS: Dict[str, List[str]] = {
    brevitas_binary_module: all_int_quantizers, 
    brevitas_scaled_int_module: all_int_quantizers,
    brevitas_ternary_module: all_ternary_quantizers, 
    brevitas_fixed_point_module: all_fp_quantizers
}
SCREEN_CHARS = 80* "-"

@dataclass
class BaseQuant():
    default_quantizer: str
    template: Optional[str] = None
    #module name, e.g. brevitas.quant.scaled_int for all int quantizers
    #does not have to be set within config
    default_quantizer_module: Optional[str] = None

    def __post_init__(self):
        #remember how often the supplied default quantizer is not within supported modules
        count: int = 0
        for quantizer_module, quantizer_classes in SUPPORTED_QUANTIZERS.items():
            if self.default_quantizer not in quantizer_classes:
                count += 1
            #remember what module in brevitas.quant the default quantizer is in (binary, fixed_point, ternary, scaled_int) 
            else: 
                self.default_quantizer_module = quantizer_module
        if count == len(list(SUPPORTED_QUANTIZERS.keys())):
            log.error(f'Quantization needs to derive from a quantizer class within modules:\n{SCREEN_CHARS}\nbrevitas.quant\n{SCREEN_CHARS}\n, not:\n{self.default_quantizer}')
            raise ValueError()
        log.debug(f'Default quantizer for {self.default_quantizer} appeared in {self.default_quantizer_module}')

#creating explicit classes in order to avoid future type collisions wiht Union[WeightQuantArgs, BiasQuantArgs, ActQuantArgs]
@dataclass
class WeightQuant(BaseQuant):
    args: Optional[WeightQuantArgs] = None

@dataclass
class BiasQuant(BaseQuant):
    args: Optional[BiasQuantArgs] = None

@dataclass
class ActQuant(BaseQuant):
    args: Optional[ActQuantArgs] = None

#map different quantization kinds to corresponding Quantizer Base Classes
MAPPINGS = {"weight": "weight", "act": "act", "bias": "bias", "input": "act", "output": "act"}

from brevitas.inject import _ExtendedInjectorType
import brevitas.quant
from importlib import import_module
from types import ModuleType

#TODO: add overriding feature for templates and args within yaml
@dataclass
class LayerQuantArgs:
    quantize: bool
    name: Optional[str] = None
    layer_type: Optional[str] = None #Linear, Embedding, MHA, ReLU ...
    #after using a default Quantizer, it is necessary to do the injection part ourselves
    #which means creating a custom Class deriving of the base brevitas quantizer class and overriding qparams
    weight: Optional[WeightQuant] = None
    bias: Optional[BiasQuant] = None
    act: Optional[ActQuant] = None
    input: Optional[ActQuant] = None
    output: Optional[ActQuant] = None

    def get_custom_quantizers(self):
        """
            Idea: Construct a custom class from the default quantizer e.g. Int8WeightPerTensorFloat and override its parameters
            with the values from the config
        """
        #mapping of custom quantizer classes for the entire layer
        quantizers: Dict[str, any] = dict()
        
        for layer_quant_name in ["weight", "bias", "act", "input", "output"]:
            quant_args = getattr(self, layer_quant_name)
            if quant_args == None:
                continue
            log.debug(f'Setting custom quantizer for type: {layer_quant_name} and args: {quant_args}')
            #get default quantizer
            quantizer_module: ModuleType = import_module(quant_args.default_quantizer_module)
            #create subclass from that quantizer and override values
            #from: https://stackoverflow.com/questions/9269902/is-there-a-way-to-create-subclasses-on-the-fly
            quantizer_args = dict()
            #type constructor needs dict for args, not (data)class
            for field in fields(quant_args.args) if quant_args.args is not None else []:
                quantizer_args[field.name] = getattr(quant_args.args, field.name)
            quantizer_class = getattr(quantizer_module, quant_args.default_quantizer)
            #make subclass of default quantizer
            #it is not an instance, but of type class
            #it also overrides qparams from default quantizer with supplied quantizers
            quantizers[layer_quant_name + '_quant'] = type(
                    'Custom' + self.name.capitalize() + self.layer_type.capitalize() + str(layer_quant_name).capitalize() + 'Quantizer', 
                    (quantizer_class,), 
                    quantizer_args
                )
        return quantizers

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
        #need to copy values as the current type of self.modules is DictConfig
        #we need a regular dict, otherwise the type of submodules is going to statically stay DictConfig
        #this is a problem when we need to access methods of e.g. LayerQuantArgs 
        modules = self.modules
        self.modules: Dict[str, Dict[str, LayerQuantArgs]] = dict()
        for module_name, module_cfg in modules.items():
            self.modules[module_name]: Dict[str, LayerQuantArgs] = dict()
            log.debug(f'Going through submodule: {module_name}')
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
                    try:
                        layer = LayerQuantArgs(**layer_cfg)
                    except:
                        log.error(f'Layer configs only support these properties: {[x.name for x in fields(LayerQuantArgs)]}. Caused by layer: {layer_name}.')
                        raise TypeError
                        #self.modules[module_name][layer_name] = layer = LayerQuantArgs(**layer_cfg)
                    if not layer.quantize:
                        continue
                    elif layer.quantize and layer.layer_type == None:
                        log.error(f'Layer {layer_name} has to contain a property \"layer_type\" which describes its type, e.g. \"Linear\" for linear layers.')
                        raise ValueError
                    #type cleanup for all defined properties
                    #goes through the supposed type of the dataclass, not of the supplied config
                    for field in (x for x in fields(layer) if x.name != 'quantize'):
                        log.debug(f'{SCREEN_CHARS}\nCleaning up field ---{field.name}--- within layer: {layer_name}')
                        attr = getattr(layer, field.name)
                        if attr == None:
                            continue
                        origin_type = get_optional_type(field.type)
                        log.debug(f'Field {field.name} within layer {layer_name} should be of type: {origin_type}, however it is of type: {type(attr)}')
                        #cleanup for nested config
                        if not isinstance(attr, origin_type) and isinstance(attr, Union[Dict, DictConfig]):
                            #attr is of type DictConfig, does not allow adding properties
                            new_attr: dict = dict(attr)
                            #cleanup for quantizer config
                            #if origin_type == BaseQuant or origin_type in QuantArgs.__subclasses__():
                            if True:
                                log.debug(f'Cleaning up {field.name} Quant arguments for layer: {layer_name}')
                                fields_key_type = {field.name:field.type for field in fields(origin_type)}
                                log.debug(f'Types to clean up: {fields_key_type}')
                                #find all properties that were passed in config (given_keys) and all other properties (difference)
                                difference = set(fields_key_type.keys()) - set(new_attr.keys())
                                given_keys = set(fields_key_type.keys()) & set(new_attr.keys())
                                #all properties that are set to None within config
                                for quant_name in difference:
                                    log.debug(f'Field {quant_name} has not been set. Defaulting to zero.')
                                    #user did not supply required parameter
                                    #if not isinstance(get_origin(fields_key_type[quant_name]), Union):
                                    if not get_origin(fields_key_type[quant_name]) is Union:
                                        log.error(f'Parameter \"{quant_name}\" for config \"{field.name}\" within layer \"{layer_cfg}\" is required.')
                                        raise KeyError
                                    new_attr[quant_name] = None
                                log.debug(new_attr)
                                log.debug(given_keys)
                                #cleanup for all properties set within config
                                for quant_name in given_keys:
                                    log.debug(f'Field {quant_name} has been supplied in yaml config. Cleaning up current instance.')
                                    quant_type = fields_key_type[quant_name]
                                    #get unwrapped type which can be called with its constructor
                                    origin_type_quant_cfg = get_optional_type(quant_type)
                                    #generic typing wrappers (Union, Optional) cannot be instantiated
                                    #problem: dataclasses allow dict unpacking (**dict). currently, entire config is passed to first param 
                                    #as it is not unpacked -> check if origin type allows **, otherwise pass value normally
                                    if hasattr(origin_type_quant_cfg, "__dataclass_fields__"):
                                        new_attr[quant_name] = origin_type_quant_cfg(**new_attr[quant_name])
                                    else:
                                        new_attr[quant_name] = origin_type_quant_cfg(new_attr[quant_name])
                            #get_args(field.type)[0](attr): call constructor of type with arguments from attr
                            #example: call list with parameters within the config 
                            setattr(layer, field.name, origin_type(**new_attr))
                            #setattr(self.modules[module_name], layer_name, layer)
                            layer.name = layer_name
                            self.modules[module_name][layer_name] = layer
                            log.debug(f'Config for field {field.name} has been cleaned up. {new_attr}')


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
        self.quant_cfg: ModelQuantArgs = quant_cfg

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
    quant_cfg = ModelQuantArgs(**_quant_cfg.model)
    log.debug(f'Configured quantization config: {quant_cfg}')
    #get_classes necessary, otherwise a circular import error will occur
    quantizer: Quantizer = get_data(log, package_self, _quant_cfg.type, Quantizer, quant_cfg)
    return quantizer