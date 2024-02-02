import torch
import math
from torch import nn 
from torch.nn import functional as F
from qtransform.model import modules as custom_nn
from dataclasses import dataclass
from typing import Optional
from brevitas.inject.defaults import Uint8ActPerTensorFloat
from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
from brevitas import nn as qnn
from brevitas.nn import utils as qutils
from brevitas.proxy import WeightQuantProxyFromInjector, BiasQuantProxyFromInjector

__all__ = ['EltwiseAdd']

class EltwiseAdd(nn.Module):
    """Layer Wrapper for torch '+' operator to Replace with qnn.QuantEltwiseAdd Fake Layer that adds two intputs together."""
    def __init__(self):
        super().__init__()

    def forward(self, input, other):
        return input + other
    

# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, normalized_shape, bias):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class BatchNorm(nn.BatchNorm1d):
    """ BatchNorm but with an optional bias and padding to support variable input length. PyTorch doesn't support simply bias=False """

    def __init__(self, num_features, bias,  *args, **kwargs): #arg names need to be identical to torch argnames for quantization support
        self.num_features = num_features
        super().__init__(num_features, *args, **kwargs)
        #self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(num_features)) if bias else None

    def forward(self, input, *args, **kwargs):
        #dirty workaround to avoid runtimeerrors by adding a padding if the input is smaller than the feature length
        #padding does not artificially lower mean as normalization is performed along the word embeddings
        n,c,l = input.size()
        if c < self.num_features:
            #input tensor should always be three dimensional
            padding = torch.zeros(n, self.num_features - c, l)
            input = torch.cat((input, padding), dim=1)
        input = super().forward(input, *args, **kwargs)
        #remove padding
        #torch.repeat not supported by FINN compiler 
        #index = torch.arange(c).reshape(c,1).repeat((n,1,l))
        #index.to(device=input.device)
        #return torch.gather(input=input, dim=1, index=index)
        return input[:,None:c]


class QuantGELU(QuantNLAL):
    """Does not work so well"""
    def __init__(
            self,
            act_quant: Optional[ActQuantType] = Uint8ActPerTensorFloat,
            input_quant: Optional[ActQuantType] = None,
            return_quant_tensor: bool = False,
            **kwargs):
        QuantNLAL.__init__(
            self,
            act_impl=nn.GELU,
            passthrough_act=True,
            input_quant=input_quant,
            act_quant=act_quant,
            return_quant_tensor=return_quant_tensor,
            **kwargs)
        


def compute_channel_view_shape(tensor: torch.Tensor, channel_dim: int):
    """
        copied from: brevitas.nn.utils.compute_channel_view_shape
    """
    broadcast_shape = [1] * len(tensor.size())
    broadcast_shape[channel_dim] = -1
    return tuple(broadcast_shape)

## TODO test this
## TODO maybe specify in args if batch norm is performed during input or output projection
def merge_bn_mha(layer, bn, output_channel_dim=0):
    #retrieve learnable parameters from batchnorm (scale + bias)
    out = qutils.mul_add_from_bn(
        bn_mean=bn.running_mean,
        bn_var=bn.running_var,
        bn_eps=bn.eps,
        bn_weight=bn.weight.data.clone(),
        bn_bias=bn.bias.data.clone())
    mul_factor, add_factor = out #scalar values
    #out_proj is QuantLinear(in_features=embd_dim, out_features=embd_dim)
    out_ch_weight_shape = qutils.compute_channel_view_shape(layer.out_proj.weight, output_channel_dim)
    #apply batchnorm during after forward pass of layer, before returning result
    layer.out_proj.weight.data.mul_(mul_factor.view(out_ch_weight_shape))
    if layer.out_proj.bias is not None:
        out_ch_bias_shape = qutils.compute_channel_view_shape(layer.out_proj.bias, channel_dim=0)
        layer.out_proj.bias.data.mul_(mul_factor.view(out_ch_bias_shape))
        layer.out_proj.bias.data.add_(add_factor.view(out_ch_bias_shape))
    else:
        layer.out_proj.bias = nn.Parameter(add_factor)
    if (hasattr(layer, 'out_proj_weight_quant') and
            isinstance(layer.out_proj_weight_quant, WeightQuantProxyFromInjector)):
        layer.out_proj_weight_quant.init_tensor_quant()
    if (hasattr(layer, 'out_proj_bias_quant') and isinstance(layer.out_proj_bias_quant, BiasQuantProxyFromInjector)):
        layer.out_proj_bias_quant.init_tensor_quant()

class LinearForBatchNorm(nn.Linear):
    """
    Very experimental linear layer that transposes a tensor of shape [N,C,L] / [C,L]
    into shape [N,L,C] / [L,C], performs linear transformation on it and then transposes it back.
    It probably is not useful as that would imply performing linear transformation on exactly one embedding
    of each word and then adding the result together.
    """
    def __init__(self, num_features: int, bias= True, *args, **kwargs):
        super().__init__(num_features, num_features, bias, *args, **kwargs)
    
    def forward(x):
        return super().forward(x.transpose(-1,-2)).transpose(-1,-2)

class CausalSelfAttention(nn.Module):
    """
    CausalSelfAttention. 
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0       
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.block_size = config.block_size
        qnn.QuantMultiheadAttention
        self.mha = nn.MultiheadAttention(config.n_embd, config.n_head, dropout=self.dropout, batch_first=True)
        self.attn_mask = torch.nn.parameter.Parameter(torch.tril(torch.ones((config.block_size,config.block_size)))) # limit to left in the input sequence
        self.flash = config.flash
        # in case we need to do attention by hand:
        if (not self.flash) or torch.__version__[3] < 2:
            log.warn("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.2")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
            self.resid_dropout = nn.Dropout(config.dropout)
        #print(self.flash)
        #print(torch.__version__[2])

    def forward(self, x):
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!HELP ME")
        # this if block is needed for toprch <2.21 where flash attention onnx export does not work

        #if not type(self.mha).__name__ == "QuantMultiheadAttention" and (not self.flash): #or torch.__version__ < (2,21):
#
        #    #log.warn("Using slower self attention for non quantized execution if torch does not support it or if flash == False")
        #    B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
        #    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        #    q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        #    k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        #    q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        #    v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        #    # manual implementation of attention
        #    att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        #    att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        #    att = F.softmax(att, dim=-1)
        #    att = self.attn_dropout(att)
        #    y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        #    y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side
#
        #    # output projection
        #    y = self.resid_dropout(self.c_proj(y))
        #else:
            #QuantMultiheadAttention does not have is_causal in constructor -> use attention mask instead
            #TODO: in QuantMultiHeadAttention, q,k,v are only transposed if param batch_first is True. Investigate
            #TODO number 2: error with incompatible sizes during forward pass in QuantMultiheadAttention
        y, weights = self.mha(x, x, x, attn_mask=self.attn_mask if self.training else None, need_weights=False) # Q, K, V, attn_mask y
            #y, weights = self.mha(x, x, x, is_causal=True) # Q, K, V, attn_mask y
        return y
from logging import getLogger
log = getLogger(__name__)

class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.active  = getattr(nn, config.transformer_active_func, None)
        if not self.active:
            log.error(f'{config.transformer_active_func} is not a valid activation function. Check property transformer_active_func')
            raise ValueError
        self.active = self.active()
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x): 
        x = self.c_fc(x)
        x = self.active(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

#TODO: either replace or merge batchnorm, or both. should be configurable in yaml files somewhere
CUSTOM_LN = False
#dirty workaround to avoid circular import error and support preprocess_for_quantize from brevitas.graph.quantize 
#qtransform.quantization.quant_bn could also be added into the fhswf-dev branch of brevitas
#TODO: wait until meeting with brevitas team to see if development in main repo will add our requirements

#from pkgutil import iter_modules
#from os.path import join
#import qtransform 
#CustomBatchNorm1d = filter(lambda x: x.name == 'quant_bn', list(iter_modules([join(qtransform.__path__[0], 'quantization')])))
from importlib import import_module
quant_bn = import_module('qtransform.quantization.quant_bn')
CustomBatchNorm1d = getattr(quant_bn, 'CustomBatchNorm1d')

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.norm_size = None
        if config.norm_layer == "LayerNorm":
            self.norm_size = config.n_embd
            #dummy layers which do nothing in order to merge with batchnorm layers
            self.custom_ln1 = nn.Identity()
            self.custom_ln2 = nn.Identity()
        elif config.norm_layer == "BatchNorm":
            self.norm_size = config.block_size
            #at the start, should do the same as quantidentity
            #after merging with batchnorm, should scale input to have a mean of 0 and a standard deviation of 1
            #TODO: should they be trainable before merging?
            self.custom_ln1 = CustomBatchNorm1d(self.norm_size, requires_grad=False) if CUSTOM_LN else nn.Identity()
            self.custom_ln2 = CustomBatchNorm1d(self.norm_size, requires_grad=False) if CUSTOM_LN else nn.Identity()
        elif config.norm_layer == "None":
            self.norm_size = None
        else:
            raise AttributeError("cannot determine model for norm layer: " + config.norm_layer)
        if self.norm_size:
            ln_1 = getattr(custom_nn, config.norm_layer, None)
            ln_2 = getattr(custom_nn, config.norm_layer, None)
            self.ln_1 = ln_1(self.norm_size, config.bias)
            self.ln_2 = ln_2(self.norm_size, config.bias)

        self.attn = CausalSelfAttention(config)
        
        self.residual1 = custom_nn.EltwiseAdd()
        self.residual2 = custom_nn.EltwiseAdd()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)

    def forward(self, x):
        if self.norm_size:
            x = self.residual1(x, self.attn(self.ln_1(x)))
            x = self.residual2(x, self.mlp(self.ln_2(x)))
            #x = x + self.attn(self.ln_1(x))
            #x = x + self.mlp(self.ln_2(x))
        else:
            x = self.residual1(x, self.attn(x))
            x = self.residual2(x, self.mlp(x))
            #x = x + self.attn(x)
            #x = x + self.mlp(x)
        return x
    