from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from torch.nn import Module as TorchModule
from brevitas.nn.mixin import * #WeightQuantType, BiasQuantType
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from typing import Optional, Union
from brevitas.quant_tensor import QuantTensor
from torch.nn import BatchNorm1d
import torch
#test if a quantized layer can be implemented which basically scales the values along a tensor and adds a bias, thereby simulating batch normalization

#TODO: maybe change tensor inplace
def check_shapes(tensor: torch.Tensor) -> torch.Tensor:
    """
    Checks if a tensor is of shape [C], [N,C] or [C,N] with N = 1 and C >= 1.
    If tensor is of a different shape, a ValueError will be thrown.
    The returning tensor will be of shape [C, 1].
    """
    shape_tensor = tensor.size()
    if len(shape_tensor) == 1:
        tensor = tensor[:,None]
    if len(shape_tensor) == 2:
        if shape_tensor[0] > 1 and shape_tensor[1] > 1:
            raise ValueError(f'Too many values to unpack for tensor {tensor}.')
        elif shape_tensor[0] == 1 and shape_tensor[1] > 1:
            tensor = tensor.transpose(0,1)
    elif len(shape_tensor) > 2:
        raise ValueError(f'Too many values to unpack for tensor {tensor}.')
    return tensor


def custom_bn1d(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """
    Forward pass of custom BatchNorm implementation. It expects a Tensor x of size [N,C] or [N,C,L]
    and both a weight and bias Tensor, each of size [C, 1] or of size [1,C] / [C].
    Each row/ embedding of a sentence (dimension C) will be multiplied with one value from the index of the corresponding
    weight tensor and added with the value of the bias tensor.
    """
    if not isinstance(x, torch.Tensor) :
        raise TypeError('Input is not a tensor')
    elif not isinstance(weight, torch.Tensor):
        raise TypeError('Weight is not a tensor')
    elif not isinstance(bias, torch.Tensor):
        raise TypeError('Bias is not a Tensor')

    weight = check_shapes(weight)
    bias = check_shapes(bias)
    #make a batch of size 1 
    if len(x.size()) == 2:
        x = x.repeat(1,1,1)
        out = x * weight + bias
        return x[0]
    #input is batched already
    return x * weight + bias

class CustomBatchNorm1d(TorchModule):
    """
    Incredibly basic implementation of Batchnorm which normalizes by scaling the input tensor with its weight and adding a bias on each element.
    """
    _weight: torch.Tensor
    _bias: torch.Tensor

    def __init__(self, num_features: int):
        if not isinstance(num_features, int) or num_features <= 0:
            raise AttributeError(f'num_features ({num_features}) not an acceptable value')
        TorchModule.__init__(self)
        self.num_features = num_features
        #do the same as identity
        self.weight = torch.ones(self.num_features)
        self.bias = torch.zeros(self.num_features)
    
    @property
    def weight(self):
        return self._weight
    
    @weight.setter
    def weight(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Cannot set weight to type {type(value)}')
        self._weight = check_shapes(value)

    @property
    def bias(self):
        return self._bias
    
    @bias.setter
    def bias(self, value: torch.Tensor) -> None:
        if not isinstance(value, torch.Tensor):
            raise TypeError(f'Cannot set bias to type {type(value)}')
        self._bias = check_shapes(value)

    def forward(self, x):
        #without merging: multiply by one, add zero 
        return custom_bn1d(x, self.weight, self.bias)
    
    def replace_bn(self, bn: BatchNorm1d, qat: bool = False):
        """
        Does the same as qtransform.quantization.quant_bn.replace_bn but for self.
        """
        replace_bn(bn=bn, new_bn=self, qat=qat)

#TODO: for some reason, CustomBatchNorm1d is more accurate to BatchNorm than the quantized version. Examinate
class QuantBatchnorm1d(QuantWBIOL, CustomBatchNorm1d):
    """
    Quantized version of CustomBatchNorm1d. The arguments for the corresponding batchnorm layer are going to be applied only during export,
    as otherwise the parameters of regular torch.nn.BatchNorm1d cannot be learned. 
    """
    def __init__(
            self,
            num_features: int,
            weight_quant: Optional[WeightQuantType] = Int8WeightPerTensorFloat,
            bias_quant: Optional[BiasQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs) -> None:
        CustomBatchNorm1d.__init__(self, num_features=num_features)
        QuantWBIOL.__init__(
            self,
            weight_quant=weight_quant,
            bias_quant=bias_quant,
            input_quant=None,
            output_quant=None,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
    
    def forward(self, input: Union[torch.Tensor, QuantTensor]) -> Union[torch.Tensor, QuantTensor]:
        return self.forward_impl(input)
    
    def inner_forward_impl(self, x: torch.Tensor, quant_weight: torch.Tensor, quant_bias: Optional[torch.Tensor]):
        #inner_forward_impl is apparently the actual forward pass of the layer 
        #multiply each row of x with one value of weight and add bias
        return custom_bn1d(x, quant_weight, quant_bias)

from brevitas.nn.utils import merge_bn
def replace_bn(bn: BatchNorm1d, new_bn: CustomBatchNorm1d = None, qat: bool = False) -> CustomBatchNorm1d:
    """
    Merges a BatchNorm layer into CustomBatchNorm1d. To do that, merge_bn from brevitas.nn.utils is used, updating the
    weights and biases of CustomBatchNorm1d. However, the newly merged instance will not be as accurate as the unquantized
    BatchNorm layer. Each normalized value has an approximate margin of error of 0.05. 
    If new_bn is omited, an instance will be created. Depending on whether qat is True or False, either QuantBatchnorm1d or 
    CustomBatchNorm1d is used.
    """
    if not isinstance(bn, BatchNorm1d):
        raise TypeError(f'Cannot merge non-batchnorm layer ({type(bn)}).')
    if not isinstance(new_bn, CustomBatchNorm1d):
        raise TypeError(f'Cannot merge batchnorm into non CustomBatchNorm1d layer (type: {type(new_bn)}).')
    if new_bn is None:
        new_bn = QuantBatchnorm1d(bn.num_features) if qat else CustomBatchNorm1d(bn.num_features)
    elif new_bn.num_features != bn.num_features:
        raise ValueError(f'Property num_features are different for new_bn ({new_bn.num_features}) and bn ({bn.num_features})')
    merge_bn(layer=new_bn, bn=bn)
    return new_bn