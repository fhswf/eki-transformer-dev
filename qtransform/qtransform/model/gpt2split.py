import logging
import os
from typing import Any, Callable
log = logging.getLogger(__name__)

import math
from dataclasses import dataclass, fields
import torch
from torch import nn as nn
from torch.nn import functional as F
from qtransform.model.modules import TransformerBlock
from qtransform.model import modules as custom_nn
from brevitas import nn as qnn

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    flash: bool = False # cuda flas hattention
    transformer_active_func: str = 'ReLU' #specify which activation function to use in MLP (feed forwad neural network)
    norm_layer: str = 'BatchNorm' # note that this is a name for a adapter module in this repository und model.modules
    single_output: bool = False # use mini runtime optimization to only predict last token, saven on some runtime but poentially currupts onnx export
    use_weight_tying: bool = True # same weights for input emb and outputi proj https://paperswithcode.com/method/weight-tying

class QTransformModelGPTMixin():
    def __init__(self, *args, **kwargs):
        super(QTransformModelGPTMixin, self).__init__(*args, **kwargs)
        pass

    @torch.no_grad()
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def check_config_options(self, model, config:GPTConfig):
        try:
            model.config = config = config if isinstance(config, GPTConfig) else GPTConfig(**config)
        except:   
            log.error(f'Model config {config} could not be applied. Config can only have options: {[x.name for x in fields(GPTConfig)]}')
        assert config.vocab_size is not None
        assert config.block_size is not None
        log.debug(f"Model config: {model.config}")

    def export(self, export_fn: Callable, sample_tensor: torch.tensor, path: str, kwargs):
        """export function returning one or more onnx models."""
        path = os.path.join(path, self.__class__.__name__ + ".onnx")
        log.info(f"using default export Method of Modelmixin, path: {path}")
        try:
            export_fn(self, sample_tensor, path, **kwargs)
        except:
            log.error(f"Export via {export_fn.__module__}.{export_fn.__name__} failed, reason", exc_info=True)
        return path
    
    def get_model_number_of_params(self):
        """calculates rounded number of params"""
        raise NotImplementedError
    
    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        """load transformer for huggingface or similar and convert it to this model"""
        raise NotImplementedError
    
class GPT2Ensemble(nn.Module, QTransformModelGPTMixin):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.check_config_options(self, config)
        # model config options
        self.use_weight_tying = config.use_weight_tying
        # layer
        self.gpt2embdedding = GPT2Embdedding(config)
        self.gpt2core = GPT2Core(config)
        self.gpt2nexttokenprediction = GPT2NextTokenPrediction(config)

        # might unneccssary but we will leave it eher for now and see if it works 
        # https://paperswithcode.com/method/weight-tying
        if self.use_weight_tying:
            self.gpt2embdedding.wte.weight = self.gpt2nexttokenprediction.linear_out.weight
    
        # init weights, apply is submudle recursive     
        self.apply(self.init_weights)
        # apply special scaled init to the residual projections to gpt2core, transformer per GPT-2 paper
        for pn, p in self.gpt2core.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

    def forward(self, x, targets=None):
        x = self.gpt2embdedding(x)
        x = self.gpt2core(x)
        x = self.gpt2nexttokenprediction(x, targets)
        return x
    
    def export(self, export_fn: Callable, sample_tensor: torch.tensor, path: str, kwargs):
        """export function returning one or more onnx models."""
        path = os.path.join(path, self.__class__.__name__ + ".onnx")
        if "split" in self.config and  self.config.split == True:
            log.info(f"Split is configured for model {self.__class__.__name__}, splitting model for export")
        log.info(f"using default export Method of Modelmixin, path: {path}")
        try:
            export_fn(self, sample_tensor, path, **kwargs)
        except:
            log.error(f"Export via {export_fn.__module__}.{export_fn.__name__} failed, reason", exc_info=True)
        return path

class GPT2Embdedding(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # config
        self.block_size = config.block_size
        # layers
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.wpe = nn.Embedding(config.block_size, config.n_embd)
        self.emb_add = custom_nn.EltwiseAdd()
        self.dropout = nn.Dropout(config.dropout)
        pass

    def forward(self, x, targets=None):
        # compute positions for pos embedding
        # TOOD option for changing batch first
        b, t = x.size()
        assert t <= self.block_size, f"Cannot forward sequence of length {t}, block size is only {self.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=x.device).unsqueeze(0) # shape (1, t)
        tok_emb = self.wte(x) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.emb_add(tok_emb, pos_emb)
        x = self.dropout(x)
        return x
    
class GPT2NextTokenPrediction(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # config options
        self.single_output = config.single_output
        # layers
        self.linear_out = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        pass

    def forward(self, x, targets=None):
        loss = None
        if targets is None and self.single_output:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.linear_out(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        else:
            # if we are given some desired targets also calculate the loss
            logits = self.linear_out(x)
            if targets is not None:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)
        return logits, loss

class GPT2Core(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        # config options
        self.norm_size = None
        
        if config.norm_layer == "LayerNorm":
            self.norm_size = config.n_embd
        elif config.norm_layer == "BatchNorm":
            self.norm_size = config.block_size
        elif config.norm_layer == "None":
            self.norm_size = None
        else:
            raise AttributeError("can determine model for norm layer: " + config.norm_layer)
        
        # layer of the transformer
        if self.norm_size:
            ln_out = getattr(custom_nn, config.norm_layer, None)
            self.transformer = nn.ModuleDict(dict(
                layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_out = ln_out(self.norm_size, config.bias),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ))

    def forward(self, x, targets=None):
        for block in self.transformer.layer:
            x = block(x)
        if self.norm_size:
            x = self.transformer.ln_out(x)
        return x
