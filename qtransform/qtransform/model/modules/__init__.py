import torch
from torch import nn 
from torch.nn import functional as F
from qtransform.model import modules as custom_nn
import math
# @torch.jit.script # good to enable when not using torch.compile, disable when using (our default)
def new_gelu(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT).
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

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
        #tensor.repeat instead of torch.tile for onnx compatibility (https://github.com/pytorch/pytorch/issues/63796)
        index = torch.arange(c).reshape(c,1).repeat((n,1,l))
        index.to(device=input.device)
        return torch.gather(input=input, dim=1, index=index)

from typing import Optional
from brevitas.inject.defaults import Uint8ActPerTensorFloat

from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
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
        
from dataclasses import dataclass
from typing import Optional


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
        # this if block is needed for toprch <2.21 where flash attention onnx export does not work
        if not type(self.mha).__name__ == "QuantMultiheadAttention" and (not self.flash) or torch.__version__ < (2,21):

            #log.warn("Using slower self attention for non quantized execution if torch does not support it or if flash == False")
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

            # output projection
            y = self.resid_dropout(self.c_proj(y))
        else:
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

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        norm_size = None
        if config.norm_layer == "LayerNorm":
            norm_size = config.n_embd
        elif config.norm_layer == "BatchNorm":
            norm_size = config.block_size
        elif config.norm_layer == "None":
            norm_size = None
        else:
            raise AttributeError("cannot determine model for norm layer: " + config.norm_layer)
        if norm_size:
            ln_1 = getattr(custom_nn, config.norm_layer, None)
            ln_2 = getattr(custom_nn, config.norm_layer, None)
        else:
            ln_1 = nn.Identity
            ln_2 = nn.Identity
        self.ln_1 = ln_1(norm_size, config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = ln_2(norm_size, config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    