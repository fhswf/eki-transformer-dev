import unittest
from qtransform.quantization import quant_bn
import torch
from dataclasses import dataclass
from logging import getLogger
from omegaconf import DictConfig
from typing import Union, Dict

log = getLogger(__name__)

@dataclass
class Args():
    batch_size: int
    block_size: int
    n_embd: int

    def __post_init_(self):
        self.batch_size = int(self.batch_size)
        self.block_size = int(self.block_size)
        self.n_embd = int(self.n_embd)

#TODO: find out how tensors can be specified in yaml files efficiently. maybe use tensors of checkpoints or from datasets / embeddings?
class CustomBatchNormTest(unittest.TestCase):
    
    custom_bn: quant_bn.CustomBatchNorm1d
    ARGS: Args
    tensor: torch.Tensor #pseudo random tensor to compare results of regular BatchNorm and CustomBatchNorm
    def setUp(self):
        #self.num_features
        if self.ARGS is None:
            log.error(f'No args specified')
        if isinstance(self.ARGS, Union[Dict, DictConfig]):
            self.ARGS = Args(**self.ARGS)
        elif not isinstance(self.ARGS, Args):
            log.error(f'Invalid type for ARGS: {type(self.ARGS)}')
        #continuously expand size of tensor to check batchnorm with variable sizes
        self.tensor = torch.randn((1, self.ARGS.n_embd))
        self.custom_bn = quant_bn.CustomBatchNorm1d(self.ARGS.block_size)

    def test_default_state(self):
        """
        Test if CustomBatchNorm1d does the same as Identity when the weight and bias attributes havent been merged with replace_bn.
        """
        size = torch.Size((self.ARGS.block_size, 1))
        #weight
        self.assertEqual(self.custom_bn.weight.size(), size)
        self.assertEqual(self.custom_bn.weight.equal(torch.ones((self.ARGS.block_size, 1))), True)
        #bias
        self.assertEqual(self.custom_bn.bias.size(), size)
        self.assertEqual(self.custom_bn.bias.equal(torch.zeros((self.ARGS.block_size, 1))), True)
        #forward pass
        log.critical(self.tensor.size())
        out = self.custom_bn(self.tensor)
        log.critical(out.size())
        self.assertEqual(out.size(), self.tensor.size())
        self.assertEqual(out.equal(self.tensor), True, f'Tensors are not the same after forward pass')

    def test_custom_bn(self):
        """
        Test shape checking and funtionality of the forward pass of CustomBatchNorm1d.
        """
        self.assertEqual(True,True)
        #empty 3d tensor
        tensor = torch.Tensor([[[]]])
        simul_weight, simul_bias = torch.randn((2,self.ARGS.block_size))
        #simulate different batch sizes (for training) and growing prompt sizes (for inference)
        for batch in range(self.ARGS.batch_size):
            tensor.cat(torch.Tensor())
            for word in range(self.ARGS.block_size):
                tensor.cat(torch.randn(1, self.ARGS.n_embd), dim=1)

    def test_replace_bn(self):
        #quant_bn.merge_bn(layer, bn)
        self.assertEqual(True,True)


    #IDEA: make a couple of optimizer steps in a model containing batchnorm, extract that, 
    #      use qtransform.quantization.quant_bn.replace_bn, compare results in terms of accuracy
    def test_accuracy(self):
        #test if params have been updated in custombatchnorm.replace_bn
        pass
        self.assertEqual(True,True)

    def runTest(self):
        self.test_default_state()

from yaml import safe_load
def suite(filename: str) -> unittest.TestSuite:
    """
        Creates test cases which each verify the correctness of CustomBatchNorm1d.
    """
    with open(filename, 'r') as file:
        args = safe_load(file)
    test = CustomBatchNormTest()
    test.ARGS = args
    return unittest.TestSuite([test])