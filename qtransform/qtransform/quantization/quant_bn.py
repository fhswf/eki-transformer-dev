from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from torch.nn import Module as TorchModule
from brevitas.nn.mixin import * #WeightQuantType, BiasQuantType
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from typing import Optional, Union
from brevitas.quant_tensor import QuantTensor
from torch import Tensor
#test if a quantized layer can be implemented which basically scales the values along a tensor and adds a bias, thereby simulating batch normalization
class QuantBatchnorm1d(QuantWBIOL, TorchModule):
    def __init__(
            self,
            num_features: int,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        TorchModule.__init__(self)
        if not isinstance(num_features, int) or num_features <= 0:
            raise AttributeError()
        #do the same as quantidentity
        self.weight = torch.ones(num_features)
        self.bias = torch.zeros(num_features)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=None,
            output_quant=None,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
    
    def forward(self, input: Union[Tensor, QuantTensor]) -> Union[Tensor, QuantTensor]:
        return self.forward_impl(input)
    
    def inner_forward_impl(self, x: Tensor, quant_weight: Tensor, quant_bias: Optional[Tensor]):
        #inner_forward_impl is apparently the actual forward pass of the layer 
        return x * quant_weight[:,None] + quant_bias[:,None]

from brevitas import nn as qnn
from brevitas.nn import utils as qutils
from brevitas.proxy import WeightQuantProxyFromInjector, BiasQuantProxyFromInjector

def merge_bn_mha(layer, bn, output_channel_dim=0):
    #retrieve learnable parameters from batchnorm (scale + bias)
    out = qutils.mul_add_from_bn(
        bn_mean=bn.running_mean,
        bn_var=bn.running_var,
        bn_eps=bn.eps,
        bn_weight=bn.weight.data.clone(),
        bn_bias=bn.bias.data.clone())
    mul_factor, add_factor = out #scalar values
    out_ch_weight_shape = qutils.compute_channel_view_shape(layer.out_proj.weight, output_channel_dim)
    #only multiply current weights with mul_factor
    layer.out_proj.weight.data.mul_(mul_factor.view(out_ch_weight_shape))
    if layer.out_proj.bias is not None:
        out_ch_bias_shape = qutils.compute_channel_view_shape(layer.out_proj.bias, channel_dim=0)
        layer.out_proj.bias.data.mul_(mul_factor.view(out_ch_bias_shape))
        layer.out_proj.bias.data.add_(add_factor.view(out_ch_bias_shape))
    else:
        layer.out_proj.bias = nn.Parameter(add_factor)
    #update quanttensor or something
    if (hasattr(layer, 'out_proj_weight_quant') and
            isinstance(layer.out_proj_weight_quant, WeightQuantProxyFromInjector)):
        layer.out_proj_weight_quant.init_tensor_quant()
    if (hasattr(layer, 'out_proj_bias_quant') and isinstance(layer.out_proj_bias_quant, BiasQuantProxyFromInjector)):
        layer.out_proj_bias_quant.init_tensor_quant()