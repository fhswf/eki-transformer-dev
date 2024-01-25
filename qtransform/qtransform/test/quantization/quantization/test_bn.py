import unittest
from qtransform.quantization import quant_bn
from torch import nn

class CustomBatchNormTest(unittest.TestCase):
    
    custom_bn: quant_bn.CustomBatchNorm1d

    def setUp(self):
        pass
        #self.num_features

    def test_default_state(self):
        """
        Test if CustomBatchNorm1d does the same as Identity when the weight and bias attributes havent been merged with replace_bn.
        """
        self.custom_bn = quant_bn.CustomBatchNorm1d(num_features)
    def test_custom_bn(self):
        """
        Test shape checking and funtionality of the forward pass of CustomBatchNorm1d.
        """
        pass

    def test_replace_bn(self):
        quant_bn.merge_bn(layer, bn)



    #IDEA: make a couple of optimizer steps in a model containing batchnorm, extract that, 
    #      use qtransform.quantization.quant_bn.replace_bn, compare results in terms of accuracy
    def test_accuracy(self):
        #test if params have been updated in custombatchnorm.replace_bn
        pass