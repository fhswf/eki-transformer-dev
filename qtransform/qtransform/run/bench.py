import logging
log = logging. getLogger(__name__)
import torch
from omegaconf import DictConfig

def run(cfg : DictConfig):
    compare_bn()

from qtransform.quantization.quant_bn import replace_bn, QuantBatchnorm1d, CustomBatchNorm1d
from qtransform.quantization.brevitas_quant import BrevitasQuantizer
from qtransform.quantization import LayerQuantConfig
"""
TODO: not sure at what step batchnorm layers should be merged.
      if batchnorm is merged before training while training from scratch, the default values would be merged 
      if batchnorm is merged before training during ptq, it would make sense but batchnorm could potentially stay in the model
      if batchnorm is merged before training during ptq, the qparams would be learned during training 
      if batchnorm is merged after training during qat, the trained values from batchnorm would be merged but the qparams would be default
      if batchnorm is merged after training during ptq, qparams would still have their default values

      based on this, it would make the most sense to merge before training during ptq
      that could make the previous quantization during qat possibly redundant

"""
def compare_bn():
    """
    Compares the custom implementation of BatchNorm1d within qtransform with torch's batchnorm.
    TODO: it would be best to pass a BatchNorm layer that had its gamma and beta tensors trained
          to have that, a transformer model would have to be trained
    """
    #inputs comparable to a small gpt2 model for fpgas during training
    n,c,l = (12,64, 256)
    size = torch.Size([n,c,l])
    torch_bn = torch.nn.BatchNorm1d(c)
    torch_bn.train()
    iters = 100
    #feed some dummy values to adjust the mean and standard deviation
    compare_loss(torch_bn, torch.nn.Identity(), size, iters)
    custom_bn = CustomBatchNorm1d(c)
    custom_bn = replace_bn(bn=torch_bn, new_bn=custom_bn, qat=False)
    #make some space for more results
    loss_fn = lambda x: compare_loss(torch_bn, custom_bn, size, 10).unsqueeze(0)
    result = compare_loss(torch_bn, custom_bn, size, 10).unsqueeze(0)
    torch_bn.eval()
    #TODO: actually benchmark this on a trained model
    for i in range(iters):
        result = torch.cat((result, compare_loss(torch_bn, custom_bn, size, 10).unsqueeze(0)), dim=0)
    log.info(f'Average loss when merging batchnorm: {result.mean()}')


def compare_loss(layer1: torch.nn.Module, layer2: torch.nn.Module, shape: torch.Size, iters: int) -> torch.Tensor:
    """
    Compares the result of two layers by passing random values with a specified shape into both layers and 
    calculating the mean from the difference of their outputs (out_layer1 - out_layer2). 
    This process is repeated iters times, returning a 1d Tensor which contains the results.

    This function should be useful for comparing the accuracy of outputs for quantized and non-quantized layers or
    custom implementations of layers such as MultiheadAttention, BatchNorm, LayerNorm etc.
    """
    result: torch.Tensor = torch.zeros(iters)
    for i in range(iters):
        #random input: TODO: maybe load dataset or something
        input = torch.randn(shape)
        out_l1 = layer1(input)
        out_l2 = layer2(input)
        result[i] = (out_l1 - out_l2).abs().mean()
    return result