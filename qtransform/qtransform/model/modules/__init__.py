import torch
from torch import nn 
from torch.nn import functional as F
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

from brevitas import nn as qnn

class CausalSelfAttention(nn.Module):
    """
    CausalSelfAttention. 
    """
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0       
        
        if config.quantize:
            self.mha = qnn.QuantMultiheadAttention(config.n_embd, config.n_head, batch_first=True)
            self.attn_mask = torch.tril(torch.ones((config.block_size,config.block_size))) # limit to left in the input sequence
        else: 
            # key, query, value projections for all heads, but in a batch
            self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
            # output projection
            self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
            # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
            self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
            if not self.flash:
                print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
                # causal mask to ensure that attention is only applied to the left in the input sequence
                self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                            .view(1, 1, config.block_size, config.block_size))

        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.quantize = config.quantize
        self.flash = config.flash
        self.block_size = config.block_size

    def forward(self, x):
        if self.quantize:
            y, weights = self.mha(x, x, x, attn_mask=self.attn_mask if self.training else None) # Q, K, V, attn_mask y
            return y
        else:
            B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

            # calculate query, key, values for all heads in batch and move head forward to be the batch dim
            q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
            k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
            v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

            # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
            if self.flash:
                # efficient attention using Flash Attention CUDA kernels
                # "scaled_dot_product_attention" does not export to onnx in all versions of pytorch , presuambly fixed in current nightly builds:
                #   https://github.com/pytorch/pytorch/issues/97262
                y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
            else:
                # # manual implementation of attention
                # att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
                # att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
                # att = F.softmax(att, dim=-1)
                # att = self.attn_dropout(att)
                # y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

                def scaled_dot_product_attention(Q,K,V, is_causal:bool, dropout_p:float, source_length, target_length, attn_mask=None):
                    # Efficient implementation equivalent to the following:
                    attn_mask = torch.ones(target_length, source_length, dtype=torch.bool).tril(diagonal=0) if is_causal else attn_mask
                    attn_mask = attn_mask.masked_fill(not attn_mask, -float('inf')) if attn_mask.dtype==torch.bool else attn_mask
                    attn_weight = torch.softmax((Q @ K.transpose(-2, -1) / math.sqrt(Q.size(-1))) + attn_mask, dim=-1)
                    attn_weight = torch.dropout(attn_weight, dropout_p)
                    return attn_weight @ V
                y  = scaled_dot_product_attention(q,k,v,is_causal=True, dropout_p=self.dropout, source_length=self.block_size, target_length=self.block_size)

            y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

            # output projection
            y = self.resid_dropout(self.c_proj(y))
            return y
        
class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.quantize = config.quantize
        if not self.quantize:
            self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
            self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
            self.activation = nn.ReLU6()
            self.active  = getattr(nn, config.transformer_active_func)()
        else:
            self.c_fc    = qnn.QuantLinear(config.n_embd, 4 * config.n_embd, bias=config.bias, weight_bit_width=8)
            self.c_proj  = qnn.QuantLinear(4 * config.n_embd, config.n_embd, bias=config.bias, weight_bit_width=8)
            self.active  = qnn.QuantReLU(bit_width=8)
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
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x
    