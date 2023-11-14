from abc import ABC, abstractclassmethod
import logging
import sys
import json
from torch.nn import Module
from omegaconf import DictConfig, OmegaConf
import pprint 
from typing import List, Optional, Dict, Tuple, Union, get_args, get_origin #get_args: get raw type of wrapper -> Optional[str] is (), Dict[str, str] = (str, str)...
from dataclasses import dataclass, fields, InitVar
from qtransform.classloader import get_data
from qtransform.utils.introspection import get_optional_type
from enum import Enum
from brevitas.inject.enum import QuantType, FloatToIntImplType, ScalingImplType, BitWidthImplType, RestrictValueType, StatsOp
from brevitas.jit import ScriptModule
import yaml
from brevitas.core import zero_point

@dataclass 
class QuantArgs:
    """
        Class for configuration parameters that can be set for either weight, act, bias, input or output config.
        The class contains all parameters that can be configured for all kinds of quantizers, however not all are going to have values
        for them. Therefore, the datatype is by default set to None. when overriding the default quantizer, the attributes of value None
        should be ignored in order to avoid injection errors with brevitas.
    """ 
    quant_type : Optional[QuantType] = None #Integer, binary, ternary, fixed point integer
    bit_width_impl_type : Optional[BitWidthImplType] = None# is the bit width backpropagated and optimised
    float_to_int_impl_type : Optional[FloatToIntImplType] = None#how should the quantized values be clipped to fit into the quantized datatype
    narrow_range : Optional[bool] = None #clip max value of data type (e.g. for 8 bits: -128:127 instead of -127:127)
    signed : Optional[bool] = None #can quantized values take on negative values
    zero_point_impl : Optional[ScriptModule] = None #how is zero point infered (static zero, from stats (weights) etc.)

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
        self.clean_types()
    
    def clean_types(self):
        """
            Clean up the data types its attributes and turn the string values into its enum representations found in brevitas.
            For example, instead of quant_type being 'INT', it should be: <QuantType.INT: 'INT'>.
            This is necessary as brevitas uses injectors and relies on its enum driven API to dynamically configure quantization options.
        """
        for field in (x for x in fields(self) if x.name != 'zero_point_impl'):
            #ignore None values
            current_value_for_field = getattr(self, field.name)
            if current_value_for_field is None:
                continue
            origin_type = get_optional_type(field.type)
            try:
                setattr(self, field.name, origin_type(current_value_for_field))
            except:
                log.warning(f'Quantization argument {field.name} can only take these values: {list(origin_type.__members__.keys())}, not \'{current_value_for_field}\' Skipping argument.')
                setattr(self, field.name, None)
        #cleanup zero_point_impl field
        if self.zero_point_impl is not None:
            if self.zero_point_impl not in zero_point.__all__:
                log.warning(f'Quantization argument zero_point_impl can only take these values: {zero_point.__all__}, not \'{self.zero_point_impl}\' Skipping argument.')
                self.zero_point_impl = None
            else: 
                self.zero_point_impl = getattr(zero_point, self.zero_point_impl, None)
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
    -> usually, quantized layers have some form of default quantizer e.g. weight for linear layers, act for activation functions, but not for all.
       usually input, output and bias quantization will not be applied as they are by default set to None
    -> if we pass a class into the quantized layer with missing qparams (quant type ...), the program will crash
    -> using default quantizers is necessary
"""

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

#structure: 'brevitas.quant.scaled_int' : [all_quantizers e.g. Int8WeightPerTensorFloat]
SUPPORTED_QUANTIZERS: Dict[str, List[str]] = {
    brevitas_binary_module: all_binary_quantizers, 
    brevitas_scaled_int_module: all_int_quantizers,
    brevitas_ternary_module: all_ternary_quantizers, 
    brevitas_fixed_point_module: all_fp_quantizers
}

from os.path import join

@dataclass
class BaseQuant():
    default_quantizer: str
    template: Optional[str] = None
    type: Optional[str] = None #weight, bias, act, input, output. if left unspecified, infer from config name
    #module name, e.g. brevitas.quant.scaled_int for all int quantizers
    #should not be set within config
    quantizer_module: str = None #InitVar[str] = None
    args: Optional[QuantArgs] = None
    
    def __post_init__(self):
        #remember how often the supplied default quantizer is not within supported modules
        failed_lookup: int = 0
        #user supplied module in which quantizer appears in, not necessary though
        if self.quantizer_module and self.quantizer_module not in SUPPORTED_QUANTIZERS.keys():
            log.warning(f'User specified that Quantizer \"{self.default_quantizer}\" appears in module \"{self.quantizer_module}\". This module is not supported.')
        for quantizer_module, quantizer_classes in SUPPORTED_QUANTIZERS.items():
            if self.default_quantizer not in quantizer_classes:
                failed_lookup += 1
            #remember what module in brevitas.quant the default quantizer is in (binary, fixed_point, ternary, scaled_int) 
            else: 
                self.quantizer_module = quantizer_module
                log.debug(f'Default quantizer for {self.default_quantizer} appeared in {self.quantizer_module}')
                break
        if failed_lookup == len(list(SUPPORTED_QUANTIZERS.keys())):
            log.error(f'Quantizer class \"{self.default_quantizer}\" did not appear in modules: {list(SUPPORTED_QUANTIZERS.keys())}')
            raise ValueError()
        #cleanup quantargs, if present
        if not isinstance(self.args, QuantArgs) and self.args is not None:
            self.args = QuantArgs(**self.args)
        #load template config
        if self.template:
            path_of_init = '/'.join(__file__.split('/')[:-1])
            #TODO: should path of template directory be configurable?
            with open(join(path_of_init,'model', 'templates', self.template + '.yaml'), 'r') as yaml_file:
                template_yaml: Dict[str, str] = yaml.safe_load(yaml_file)
            #currently, arguments from model yaml file are loaded in self.args
            #the existing values should override values from template
            #therefore, only values not set in self.args attribute are looked at
            template_args = template_yaml.get('args')
            if not template_args:
                log.warning(f'Template {self.template} is missing property \"args\" which contains the quant parameters. Skipping template configuration.')
                return
            #all possible config options in self.args which are not set in config
            empty_qparams = set([x.name for x in fields(self.args) if getattr(self.args, x.name, None) is None])
            #only apply values from template which are not currently set and which are also supported
            for field in set(template_args.keys()) & empty_qparams :
                setattr(self.args, field, template_args[field])
        
        
        
#creating explicit classes in order to avoid future type collisions with Union[WeightQuantArgs, BiasQuantArgs, ActQuantArgs]
#TODO: make BaseQuant contain generic Type of args e.g. BaseQuant[WeightQuantArgs]
@dataclass
class WeightQuant(BaseQuant):
    args: Optional[WeightQuantArgs] = None

@dataclass
class BiasQuant(BaseQuant):
    args: Optional[BiasQuantArgs] = None

@dataclass
class ActQuant(BaseQuant):
    args: Optional[ActQuantArgs] = None

from importlib import import_module
from types import ModuleType

from re import search, subn
import qtransform.quantization as package_self

@dataclass
class LayerQuantConfig:
    quantize: bool = True #if layer is specified in config, assume that it should be quantized
    layer_type: Optional[str] = None #Linear, Embedding, MHA, ReLU ...
    quantized_layer_class: type = None
    name: str = None #necessary to iterate through layers in model. set by ModelQuantArgs __post_init__
    #after using a default Quantizer, it is necessary to do the injection part ourselves
    #which means creating a custom Class deriving of the base brevitas quantizer class and overriding qparams
    quantizers: Optional[Dict[str, BaseQuant]] = None #generic container for all quantizers

    def __post_init__(self):
        #check if layer should even be quantized
        try:
            self.quantize = bool(self.quantize)
        except:
            log.error(f'Property \"quantize\" of layer {self.name} should be True or False, not {self.quantize}')
            raise TypeError
        if self.quantize and self.layer_type == None:
            log.error(f'Layer {self.name} has to contain a property \"layer_type\" which describes its type, e.g. \"Linear\" for linear layers.')
            raise TypeError
        if not isinstance(self.name, str):
            try:
                self.name = str(self.name)
            except TypeError:
                log.error(f'Name for layer \"{self.name}\" should be of type string, not {type(self.name)}')
                raise TypeError

        #cleanup layer config for non-quantizer specific properties
        for field in (x for x in fields(self) if x.name not in ['quantized_layer_class', 'quantizers']):#['quantize', 'quantized_layer_class', 'layer_type']):
            if hasattr(log,"trace"): log.trace(f"Cleaning up field:  {field.name:10s}\t within layer: {self.name}")
            attr = getattr(self, field.name)
            if attr == None:
                if not get_origin(field.type) is Union:
                    log.error(f'Parameter \"{field.name}\" for config \"{field.name}\" within layer \"{self.name}\" is required.')
                    raise KeyError
                else: 
                    log.debug(f'Field {field.name} has not been set. Defaulting to zero.')
                    continue
            origin_type = get_optional_type(field.type) #unwrap Optional[origin_type] to origin_type for constructor
            log.debug(f'Field {field.name} within layer {self.name} should be of type: {origin_type}, however it is of type: {type(attr)}')
            #typecheck current type of value, not supposed value
            try:
                #check if config attribute is of type dict or is a dataclass which supports dict unpacking
                if isinstance(attr, Union[Dict, DictConfig]):
                    #Unwrap generic typing classes
                    _origin_type = get_origin(origin_type) 
                    origin_type = _origin_type if _origin_type is not None else origin_type
                    setattr(self, field.name, origin_type(**attr))
                else:
                    setattr(self, field.name, origin_type(attr))
            except TypeError as e:
                #TODO: Logfiles for errors
                log.error(f'Something went wrong with layer {self.name} while processing argument {field.name}. Reason: {e}')

        #cleanup quantizers
        #currently, self.quantizers is read-only DictConfig from Hydra 
        quantizers = self.quantizers
        self.quantizers = dict()
        for quantizer_name, quantizer_cfg in quantizers.items() if quantizers is not None else []:
            if hasattr(log,"trace"): log.trace(f"Cleaning up quantizer:  {quantizer_name:10s}\t within layer: {self.name}")
            if not isinstance(quantizer_cfg, Union[Dict, DictConfig]):
                log.error(f'Quantizer {quantizer_name:10s}\t within layer: {self.name} is not nested.')
            #get type of quantizer in order to construct specific quantargs instance
            assumed_quantizer_type = quantizer_cfg.get('type', default_value="")
            if assumed_quantizer_type == "": #property not specified
                log.debug(f'Field "type" for quantizer {quantizer_name} of layer {self.name} not specified. Assuming type from quantizer name.')
                assumed_quantizer_type = search(r'(weight|act|bias|input|output)', quantizer_name)
                assumed_quantizer_type = assumed_quantizer_type.group(0) if assumed_quantizer_type else ""
            #input and output type are activations
            assumed_quantizer_type: str = subn(r'(input|output)', 'act', assumed_quantizer_type)[0]
            if assumed_quantizer_type not in ('weight', 'act', 'bias'):
                log.error(f'Type of Quantizer \"{quantizer_name}\" for layer {self.name} can not be categorized as either weight, act or bias')
                raise ValueError
            #use first type in name
            quantizer_class = getattr(package_self, assumed_quantizer_type.capitalize() + 'Quant')
            try:
                self.quantizers[quantizer_name] = quantizer_class(**{"type": assumed_quantizer_type[0], **quantizer_cfg})
            except Exception as e:
                log.error(f'Quantizer cleanup for quantizer {quantizer_name} of layer {self.name} failed due to: {e}')

    def get_layers(self) -> Tuple[str]:
        """
            Splits the name of the quantization config for this layer into a list in order to iterate through a model.
            Usually, the property name of LayerQuantConfig is presented in dotted form, each entry being a sublayer of the model.
            E.g. transformer.layer.1.attn.mha: 
            The last property (in the case of the example: mha) is the layer for which the config should be applied.
        """
        return tuple(self.name.split('.')) if self.name else ()

    def get_custom_quantizers(self) ->Dict[str, type]:
        """
            Retrieves default quantizers of a layer and overrides their qparams with specified qparams in the config, if present. It does this by retrieving the
            quantizer that should be used for the corresponding layer from the property 'default_quantizer' of the quantizer config. This property is mandatory in order to 
            support custom quantization. If no custom qparams for certain layers are set, the qparams of the quantizer specified in 'default_quantizer' is used. 
            Otherwise, a custom quantizer is created and qparams specified in the quantization config are used. If no default quantizer is specified, the quantizer of
            the layer is used. Usually, if that is the case, only quantization for weights are applied due to the implementation of brevitas.

            Returns: Dict[quantization_kind: custom_quantizer_class] e.g. {"weight": <class Int8WeightTensorFloat>, "bias": <class CustomBiasLinearQuantizer>} 
                     if default quantizers for the corresponding quantization_kind is set. Otherwise {}
        """
        #mapping of custom quantizer classes for the entire layer
        quantizers: Dict[str, type] = dict()
        
        for quantizer_name, quantizer_cfg in self.quantizers.items() if self.quantizers is not None else []:
            log.debug(f'Setting custom quantizer for type: {quantizer_name} and args: {quantizer_cfg}')
            #import module that has default quantizer
            quantizer_module: ModuleType = import_module(quantizer_cfg.quantizer_module)
            #create subclass from that quantizer and override values
            #from: https://stackoverflow.com/questions/9269902/is-there-a-way-to-create-subclasses-on-the-fly
            quantizer_args = dict()
            #type constructor needs dict for args, not (data)class
            for field in fields(quantizer_cfg.args) if quantizer_cfg.args is not None else []:
                quant_value = getattr(quantizer_cfg.args, field.name)
                if quant_value:
                    quantizer_args[field.name] = quant_value
            #property access of class from module
            quantizer_class = getattr(quantizer_module, quantizer_cfg.default_quantizer)
            #cleanup name of quantizer for quantized brevitas class
            #structure: <type>_quant
            suffix = '_quant' if search('_quant', quantizer_name) is None else ''
            #make subclass of default quantizer
            #it is not an instance, but of type class
            #it also overrides qparams from default quantizer with supplied quantizers
            quantizers[quantizer_name + suffix] = type(
                    'Custom' + self.name.capitalize() + self.layer_type.capitalize() + str(quantizer_name).capitalize() + 'Quantizer', 
                    (quantizer_class,), 
                    quantizer_args
                )
        return quantizers

@dataclass
class ModelQuantConfig:
    cls: str
    layers: Dict[str, LayerQuantConfig]
    dtype: str
    device: str

    def __post_init__(self):
        """
            Check if the types are correct in order to prevent future issues with Brevitas. To do so,
            it iterates through the entire dict representation of the yaml config file and creates instances the corresponding
            dataclasses if necessary. For example, if a module is not of type LayerQuantConfig, the method creates an instance of
            LayerQuantConfig with the parameters supplied in the current version of the object.
        """
        if not isinstance(self.layers, Union[Dict, DictConfig]):
            log.error(f'Model config has to contain a dictionary of quantized submodules, not type: {type(self.layers)}.')
            raise TypeError
        #need to copy values as the current type of self.modules is DictConfig
        #we need a regular dict, otherwise the type of submodules is going to statically stay DictConfig
        #this is a problem when we need to access methods of e.g. LayerQuantConfig
        layers = self.layers
        if hasattr(log, "trace"): log.trace(f"ModelQuantConfig modules: {self.layers}")
        self.layers: Dict[str, LayerQuantConfig] = dict()
        #submodules_list_string contains the order of layers preceding the layer that has to be quantized
        #seperated with dots e.g. transformer.layer.1.attn.mha
        #layer_cfg is the quantization config for the last layer within submodules_list_string, so for the example
        #it would be mha
        for submodules_list_string, layer_cfg in layers.items():
            if not isinstance(layer_cfg, Union[Dict, DictConfig]):
                log.error(f'Config for layer \"{submodules_list_string}\" has to be a dictionary, not {type(layer_cfg)}')
                raise TypeError
            elif layer_cfg.get('quantize') != True and layer_cfg.get('quantize') is not None:
                continue
            #use last config of layer that is mentioned multiple times in yaml file
            elif submodules_list_string in self.layers.keys():
                log.warning(f"""Config for layer {submodules_list_string} already exists with properties \n{self.layers[submodules_list_string]}\n. Replacing them with {pprint.PrettyPrinter(indent=1).pformat(layer_cfg)}""")
            if hasattr(log, "trace"): log.trace(f"Processing layer \"{submodules_list_string}\" {layer_cfg}")
            #quick check if properties in config (do not) appear in LayerQuantConfig dataclass
            layer = LayerQuantConfig(**{"name": submodules_list_string, **layer_cfg})
            try:
                layer = LayerQuantConfig(**{"name": submodules_list_string, **layer_cfg})
            except TypeError as e:
                log.error(f'Layer configs only support these properties: {[x.name + " (required)" if get_origin(x.type) is not Union else x.name for x in fields(LayerQuantConfig)]}. Caused by layer: {submodules_list_string}.')
                raise TypeError
            self.layers[submodules_list_string] = layer

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

    def __init__(self, quant_cfg: ModelQuantConfig):
        self.quant_cfg: ModelQuantConfig = quant_cfg

    @abstractclassmethod
    def get_quantized_model(self, model: Module, inplace: bool = False) -> Module:
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

def get_quantizer(_quant_cfg: DictConfig) -> Quantizer:
    if hasattr(log,"trace"): log.trace("launched with config: " + json.dumps(OmegaConf.to_container(_quant_cfg), indent=2))
    quant_cfg = ModelQuantConfig(**_quant_cfg.model)
    if hasattr(log,"trace"): log.trace(f'Configured quantization config: {pprint.PrettyPrinter(indent=1).pformat(quant_cfg)}')
    #get_classes necessary, otherwise a circular import error will occur
    quantizer: Quantizer = get_data(log, package_self, _quant_cfg.type, Quantizer, quant_cfg)
    sys.exit(1000)
    return quantizer