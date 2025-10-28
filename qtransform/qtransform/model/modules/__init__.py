import torch
import math
from torch import Tensor, nn 
from torch.nn import functional as F
from qtransform.model import modules as custom_nn
from dataclasses import dataclass
from typing import Optional, Union
from brevitas.inject.defaults import Uint8ActPerTensorFloat
from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
from qtransform import device_singleton
from brevitas import nn as qnn
from brevitas.quant_tensor import QuantTensor

__all__ = ['EltwiseAdd', 'SinPosEmb', 'BinPosEmb']


def unpack_from_quant(tensor: torch.Tensor | QuantTensor):
    """ Unpacks the standard PyTorch tensor from a brevitas QuantTensor if applicable"""
    if isinstance(tensor, QuantTensor):
        return tensor.value
    return tensor

class EltwiseAdd(nn.Module):
    """Layer Wrapper for torch '+' operator to Replace with qnn.QuantEltwiseAdd Fake Layer that adds two intputs together."""
    def __init__(self):
        #self.to(device_singleton.device)
        super().__init__()

    def forward(self, input, other):
        return input + other
    
class SinPosEmb(nn.Module):
    """Sinosoidal Position Embedding. Does not add anything, only computes sin/cos embbedding, basically returns a constant."""
    def __init__(self, max_seq_len: int, emb_dim__model: int):
        """
        Arguments:
            emb_dim__model: int, vocab embedding dimensionality
            max_seq_len: int, maximum sequence length
        """
        super().__init__()
        # TODO in/out.py TEsting during export
        print(f"emb_dim__model {emb_dim__model}", f"max_seq_len {max_seq_len}")
        pos_id = torch.as_tensor([[n] for n in range(max_seq_len)])
        freq_div = torch.as_tensor([1e4 ** -(i / emb_dim__model) for i in range(0, emb_dim__model, 2)])
        pe = torch.zeros(max_seq_len, emb_dim__model)
        pe[:, 0::2] = torch.sin(pos_id * freq_div)
        pe[:, 1::2] = torch.cos(pos_id * freq_div)
        self.register_buffer('pe', pe)

    def forward(self, x) -> Tensor:
        """x is unused for computation"""
        return self.pe
    
class BinPosEmb(torch.nn.Module):
    """Position Embedding as encoded onehot Vector. Does not add anything, returns a constant."""
    def __init__(self, max_seq_len: int, emb_dim__model: int):
        """
        Arguments:
            emb_dim__model: int, vocab embedding dimensionality
            max_seq_len: int, maximum sequence length
        """
        super().__init__()
        pe = torch.as_tensor([
            [(n & (1 << bit)) >> bit for bit in range(emb_dim__model)] for n in range(max_seq_len)
        ])
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x is unused for computation"""
        #   Note: Convert encoding to bipolar representation
        return 2 * self.pe - 1


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
        #self.bias = nn.Parameter(torch.zeros(num_features)) if bias else None

    def forward(self, input, *args, **kwargs):
        #dirty workaround to avoid runtimeerrors by adding a padding if the input is smaller than the feature length
        #padding does not artificially lower mean as normalization is performed along the word embeddings
        n,c,l = input.size()
        #print(n,c,l)
        if c < self.num_features:
            #input tensor should always be three dimensional
            padding = torch.zeros(n, self.num_features - c, l).to(device=input.device)
            input = torch.cat((input, padding), dim=1)
            del padding
        input = super().forward(input, *args, **kwargs)
        #remove padding
        #input tensor of shape [N,C,L] gets padded to shape [N, F, L] (F >= C) and then unpadded to shape [N,C,L] 
        return input[:,None:c]

class BatchNormIdNoReplace(nn.Module):
    """ Like our custom BatchNorm but with an Ident """
    def __init__(self, num_features, bias, *args, **kwargs): #arg names need to be identical to torch argnames for quantization support
        super().__init__(*args, **kwargs)
        self.bn = BatchNorm(num_features, bias, *args, **kwargs)
        self.id = nn.Identity()
        
    def forward(self, input, *args, **kwargs):
        return self.id(self.bn(input, *args, **kwargs))

class BatchNormIdPure(nn.BatchNorm1d):
    """ Like our custom BatchNorm but with an Ident """
    def __init__(self, num_features, bias, *args, **kwargs): #arg names need to be identical to torch argnames for quantization support
        self.num_features = num_features
        super().__init__(num_features, *args, **kwargs)
        #self.weight = nn.Parameter(torch.ones(ndim))
        #self.bias = nn.Parameter(torch.zeros(num_features)) if bias else None
        self.id = nn.Identity()
        
    def forward(self, input, *args, **kwargs):
        return self.id(super().forward(input, *args, **kwargs))


class BatchNormTranspose(nn.Module):
    """ BatchNorm with Transposed Inputs to compute Norm over the last dimension."""
    def __init__(self, num_features, bias, *args, **kwargs): #arg names need to be identical to torch argnames for quantization support
        super().__init__(*args, **kwargs)
        self.bn = torch.nn.BatchNorm1d(num_features)
        #self.bn.bias = nn.Parameter(torch.zeros(num_features)) if bias else None
        self.id = nn.Identity()
        
    def forward(self, input, *args, **kwargs):
        x = self.bn(torch.transpose(input, dim0=-1, dim1=-2))
        return self.id(torch.transpose(x, dim0=-1, dim1=-2))

class InstanceNorm(nn.InstanceNorm1d):
    """
    Boilerplate class to enable specifying InstanceNorm instead of InstanceNorm1d
    """
    def __init__(self,num_features, eps=1e-5, momentum=0.1, affine=False, track_running_stats=False):
        super().__init__(num_features=num_features, eps=eps, momentum=momentum, affine=affine, track_running_stats=track_running_stats)

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
        # i think this is alerady part of mha, so we dont need this, at least if we want to replicate gpt2 :
        #self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        #since synthesis cannot use is_causal due to bool dtype and brevitas mha does not have is_causal in forward pass function
        bias = torch.ones(config.block_size, config.block_size).tril(diagonal=0)
        #torch.mha uses torch.nn.functional.scaled_dot_product_attention, attention mask is added (plus operation) to attention
        #meaning: explicitly use negative infinity to prevent adding zero instead of masking that value during softmax
        bias = bias.masked_fill(bias == 0, float('-inf'))
        #attention is added to tensor
        bias = bias.masked_fill(bias == 1, 0)
        self.register_buffer("bias", bias)

        # TODO figure out we still need this:

        # in case we need to do attention by hand:
        #if (not self.flash) or torch.__version__[3] < 2:
        #    log.warn("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.2")
        #    # causal mask to ensure that attention is only applied to the left in the input sequence
        #    self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        #    self.attn_dropout = nn.Dropout(config.dropout)
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
            #TODO: with torch version 2.0, output becomes nan when model is in eval mode. outside of CausalSelfAttention, mha does not return nan
            #      tensors during eval mode
            #due to inference, input features can be lower than specified max context length which causes problem during attention calculation
            y, weights = self.mha(x, x, x, attn_mask=self.bias[:C,:C], need_weights=False) # Q, K, V, attn_mask y
        #y = self.resid_dropout(self.c_proj(y))
        y = self.resid_dropout(y)
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
        self.residual1 = custom_nn.EltwiseAdd()
        self.residual2 = custom_nn.EltwiseAdd()
        self.attn = CausalSelfAttention(config)
        self.mlp = MLP(config)
        #necessary for quantized batchnorm to create quanttensor from tensor
        self.identity = torch.nn.Identity()
        
        self.norm_size = None
        if config.norm_layer == "BatchNormTranspose":
            self.norm_size = config.n_embd
        elif config.norm_layer == "BatchNormIdPure":
            self.norm_size = config.block_size
        elif config.norm_layer == "BatchNormIdNoReplace":
            self.norm_size = config.block_size
        elif config.norm_layer == "LayerNorm":
            self.norm_size = config.n_embd
            #dummy layers which do nothing in order to merge with batchnorm layers
            #that also means including some bloat layers
            self.custom_ln1 = nn.Identity()
            self.custom_ln2 = nn.Identity()
        elif config.norm_layer in ["BatchNorm", "InstanceNorm"]:
            self.norm_size = config.block_size
        elif config.norm_layer == "None":
            self.norm_size = None
        else:
            raise AttributeError("cannot determine model for norm layer: " + config.norm_layer)
    
        if self.norm_size:
            ln_1 = getattr(custom_nn, config.norm_layer, None)
            ln_2 = getattr(custom_nn, config.norm_layer, None)
            self.ln_1 = ln_1(self.norm_size, config.bias)
            self.ln_2 = ln_2(self.norm_size, config.bias)
        # dummy ident for block entry
        self.in_id = nn.Identity()
        
        self.attn = CausalSelfAttention(config)

    def forward(self, x):
        # Ident for possible quantization before resudials or norm layers?
        x = self.in_id(x)

        if self.norm_size:
            x = self.ln_1(x)
            x = self.residual1(x, self.attn(x))
            x = self.ln_2(x)
            x = self.residual2(x, self.mlp(x))
            #x = x + self.attn(self.ln_1(x))
            #x = x + self.mlp(self.ln_2(x))
        else:
            x = self.residual1(x, self.attn(x))
            x = self.residual2(x, self.mlp(x))
            #x = x + self.attn(x)
            #x = x + self.mlp(x)
        return x
    