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
            padding = torch.zeros(n, self.num_features - c, l).to(device=input.device)
            input = torch.cat((input, padding), dim=1)
            del padding
        input = super().forward(input, *args, **kwargs)
        #remove padding
        #input tensor of shape [N,C,L] gets padded to shape [N, F, L] (F >= C) and then unpadded to shape [N,C,L] 
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

#TODO: torch attention generates poor results during inference, karpathy's mha from scratch does not.

class KarpathyCausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side"""
        y, weights = self.mha(x, x, x, attn_mask=self.attn_mask if self.training else None, need_weights=False) # Q, K, V, attn_mask y
        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

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
        self.flash = config.flash
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        #since synthesis cannot use is_causal due to bool dtype and brevitas mha does not have is_causal in constructor
        bias = torch.ones(config.block_size, config.block_size).tril(diagonal=0)
        #torch.mha uses torch.nn.functional.scaled_dot_product_attention, attention mask is added (plus operation) to attention
        #meaning: explicitly use negative infinity to prevent adding zero instead of masking that value during softmax
        bias = bias.masked_fill(bias == 0, float('-inf'))
        self.register_buffer("bias", bias)
        # in case we need to do attention by hand:
        if (not self.flash) or torch.__version__[3] < 2:
            log.warn("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.2")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            self.attn_dropout = nn.Dropout(config.dropout)
        #print(self.flash)
        #print(torch.__version__[2])

    def forward(self, x):
        #print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!HELP ME")
        # this if block is needed for toprch <2.21 where flash attention onnx export does not work

        '''if not type(self.mha).__name__ == "QuantMultiheadAttention" and (not self.flash): #or torch.__version__ < (2,21):
        
            log.warn("Using slower self attention for non quantized execution if torch does not support it or if flash == False")
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)
            #    # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            #    # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side'''
        if False:
            pass
        else:
            #QuantMultiheadAttention does not have is_causal in constructor -> use attention mask instead
            """
            from scaled dot product attention:
                L: context, S: embedding dimension
                L, S = query.size(-2), key.size(-2)
                scale_factor = 1 / math.sqrt(query.size(-1)) if scale is None else scale
                attn_bias = torch.zeros(L, S, dtype=query.dtype)
                if is_causal:
                    assert attn_mask is None
                    temp_mask = torch.ones(L, S, dtype=torch.bool).tril(diagonal=0)
                    attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
                    attn_bias.to(query.dtype)

            """
            N, C, L = x.size()
            #due to inference, input features can be lower than specified max context length which causes problem during attention calculation
            y, weights = self.mha(x, x, x, attn_mask=self.bias[:C,:C], need_weights=False, is_causal=False) # Q, K, V, attn_mask y
        y = self.resid_dropout(self.c_proj(y))
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

#dirty workaround to avoid circular import error and support preprocess_for_quantize from brevitas.graph.quantize 
#qtransform.quantization.quant_bn could also be added into the fhswf-dev branch of brevitas
#TODO: wait until meeting with brevitas team to see if development in main repo will add our requirements
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
            #that also means including some bloat layers
            self.custom_ln1 = nn.Identity()
            self.custom_ln2 = nn.Identity()
        elif config.norm_layer == "BatchNorm":
            self.norm_size = config.block_size
            #should do the same as quantidentity as long as requires_grad is set to False
            #after merging with batchnorm, should scale input to have a mean of 0 and a standard deviation of 1
            #TODO: should they be trainable before/ after merging?
            #      layers are not moved to cuda with model.to(device). for now, workaround by specifying device in constructor
            self.custom_ln1 = CustomBatchNorm1d(self.norm_size, requires_grad=False) if config.custom_ln else nn.Identity()
            self.custom_ln2 = CustomBatchNorm1d(self.norm_size, requires_grad=False) if config.custom_ln else nn.Identity()
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
            x = self.custom_ln1(x)
            x = self.residual1(x, self.attn(self.ln_1(x)))
            x = self.custom_ln2(x)
            x = self.residual2(x, self.mlp(self.ln_2(x)))
            #x = x + self.attn(self.ln_1(x))
            #x = x + self.mlp(self.ln_2(x))
        else:
            x = self.residual1(x, self.attn(x))
            x = self.residual2(x, self.mlp(x))
            #x = x + self.attn(x)
            #x = x + self.mlp(x)
        return x
    