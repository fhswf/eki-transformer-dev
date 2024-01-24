from brevitas.nn.quant_layer import QuantWeightBiasInputOutputLayer as QuantWBIOL
from torch.nn import Module as TorchModule
from brevitas.nn.mixin import * #WeightQuantType, BiasQuantType
from brevitas.quant.scaled_int import Int8WeightPerTensorFloat
from typing import Optional, Union
from brevitas.quant_tensor import QuantTensor
from torch import Tensor
from torch.nn import BatchNorm1d
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
        self.num_features = num_features
        self.weight = torch.ones(self.num_features)
        self.bias = torch.zeros(self.num_features)
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
        #multiply each row of x with one value of weight and add bias
        return x * quant_weight[:,None] + quant_bias[:,None]

from brevitas.nn.utils import merge_bn
def merge_quant_bn(bn: BatchNorm1d, quant_bn: QuantBatchnorm1d = None) -> QuantBatchnorm1d:
    """
    Merges a BatchNorm layer into QuantBatchnorm1d. To do that, merge_bn from brevitas.nn.utils is used, updating the
    weights and biases of QuantBatchnorm1d. The merged QuantBatchnorm1d instance will not be as accurate as the unquantized
    BatchNorm layer however, with each normalized value having an approximate margin of error of 0.05. 
    If quant_bn is omited, an instance of QuantBatchnorm1d will be created and returned.
    """
    if not isinstance(bn, BatchNorm1d):
        raise TypeError(f'Cannot merge non-batchnorm layer ({type(bn)}).')
    if quant_bn is None:
        quant_bn = QuantBatchnorm1d(bn.num_features)
    elif quant_bn.num_features != bn.num_features:
        raise ValueError(f'Property num_features are different for quant_bn ({quant_bn.num_features}) and bn {bn.num_features}')
    merge_bn(layer=quant_bn, bn=bn)
    return quant_bn