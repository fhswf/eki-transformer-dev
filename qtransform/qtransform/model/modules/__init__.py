import torch
import math
from torch import nn 
from torch.nn import functional as F
from qtransform.model import modules as custom_nn
from dataclasses import dataclass
from typing import Optional, Union
from brevitas.inject.defaults import Uint8ActPerTensorFloat
from brevitas.nn.quant_layer import ActQuantType
from brevitas.nn.quant_layer import QuantNonLinearActLayer as QuantNLAL
from brevitas import nn as qnn
from brevitas.quant_tensor import QuantTensor

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
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)
        #since synthesis cannot use is_causal due to bool dtype and brevitas mha does not have is_causal in forward pass function
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
            #self.custom_ln1 = nn.Identity()
            #self.custom_ln2 = nn.Identity()
        elif config.norm_layer in ["BatchNorm", "InstanceNorm"]:
            self.norm_size = config.block_size
            #self.custom_ln1 = CustomBatchNorm1d(self.norm_size, requires_grad=False) if config.custom_ln else nn.Identity()
            #self.custom_ln2 = CustomBatchNorm1d(self.norm_size, requires_grad=False) if config.custom_ln else nn.Identity()
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
        #necessary for quantized batchnorm to create quanttensor from tensor
        self.identity = torch.nn.Identity()

    def quant_bn_forward(self, batch: Union[torch.Tensor, QuantTensor], ln: qnn.BatchNorm1dToQuantScaleBias):
        """
        Passes a batch of dimension [N,C,L] to BatchNorm1dToQuantScaleBias which expects tensors of size [C,L].
        Each sample of the batch is normalized and then concatted into a tensor of the original shape. 
        TODO: unsure if this should be placed in forward pass of BatchNorm1dToQuantScaleBias in our brevitas fork or here
        """
        if not isinstance(ln, qnn.BatchNorm1dToQuantScaleBias):
            raise TypeError(f'Passed quantized BatchNorm layer is of type {type(ln)}, not BatchNorm1dToQuantScaleBias')
        #batch is one sample
        if len(batch.size()) == 2:
            return ln(batch).unsqueeze(0)
        #dropout layer (from gpt model) returns quanttensor instead of regular tensor, iterating does not work with quanttensor
        #TODO: unsure if this would break the qonnx export
        if isinstance(batch, QuantTensor):
            device = batch.value.device
            N,C,L = batch.size()
            samples = batch.value.chunk(N) #unsure if inplace forward pass of quanttensor value will lead to undefined behavior
            out = torch.zeros(N,C,L).to(device)
        else:
            #samples = batch.chunk(batch.size()[0])
            #out = torch.zeros(batch.size())
            #in place modification of tensors
            samples = batch
            out = samples
        for i, sample in enumerate(samples):
            #chunking batch returns input of shape [1, C, L], BatchNorm1dToQuantScaleBias then returns tensor of shape [1,C,C,L]
            #first dimension can be squeezed together, tensors of second dimension are all the same
            #therefore, take the first "batch"
            """
            TODO: when out tensor is set to device, this error occurs: 
            
                RuntimeError: Output 1 of UnbindBackward0 is a view and its base or another view of its base has been modified inplace. 
                This view is the output of a function that returns multiple views. Such functions do not allow the output views to be modified inplace. 
                You should replace the inplace operation by an out-of-place one.
            """
            x = ln(sample)
            #quanttensor value is read only
            out[i] = x.squeeze(dim=0)[0] if isinstance(x, torch.Tensor) else x.value.squeeze(dim=0)[0]
        #dirty workaround to make batchnorm work, since output tensor is not quantized yet
        if isinstance(self.identity, torch.nn.Identity):
            self.identity = qnn.QuantIdentity()
            #somehow infer on which device model should be
            self.identity.to(device=self.attn.bias.device)
        out.to(samples[0].device)
        out = self.identity(out)
        return out
    def forward(self, x):
        if self.norm_size:
            #x = self.custom_ln1(x)
            #brevitas batchnorm does not support batches
            #TODO: unsure if brevitas batchnorm applies padding for smaller inputs
            if isinstance(self.ln_1, qnn.BatchNorm1dToQuantScaleBias):
                x = self.quant_bn_forward(x, self.ln_1)
            else:
                x = self.ln_1(x)
            x = self.residual1(x, self.attn(x))
            #x = self.custom_ln2(x)
            if isinstance(self.ln_2, qnn.BatchNorm1dToQuantScaleBias):
                x = self.quant_bn_forward(x, self.ln_2)
            else:
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
    