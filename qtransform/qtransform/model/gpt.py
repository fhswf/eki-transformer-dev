import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F
from qtransform.model.modules import LayerNorm, TransformerBlock
from qtransform.model import modules as custom_nn
from brevitas import nn as qnn
import logging
log = logging.getLogger(__name__)

from abc import ABC, abstractmethod
class Model(ABC):
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """ foward pass, generating num_tokens"""    
        raise NotImplementedError
    def get_num_params(self):
        """ get the number of trainable model params """
        raise NotImplementedError
    @abstractmethod
    def get_config(self):
        """ return model config """
        raise NotImplementedError
    def estimate_size(self):
        """
        Get an estimate of the raw size of the model. 
        This does not refelct the required size during training and might also not be accurate in your hardware.
        """
        raise NotImplementedError
    
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
    custom_ln: bool = False #use CustomBatchNorm1d before BatchNorm
    use_causal: bool = False
    shift_targets: bool = False # True: labels are shifted by one to the right inside the model, False: shifting is done by dataloader

from dataclasses import fields
class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        try:
            self.config = config = config if isinstance(config, GPTConfig) else GPTConfig(**config)
            log.debug(f'Applied config: \n{self.config}')
        except:   
            log.error(f'Model config \n{config}\n could not be applied. Config can only have options: {[x.name for x in fields(GPTConfig)]}')
        assert config.vocab_size is not None
        assert config.block_size is not None
        log.info(f"Model config: {self.config}")
        
        self.single_output = config.single_output
        self.use_weight_tying = config.single_output
        self.norm_size = None

        if config.norm_layer == "LayerNorm":
            self.norm_size = config.n_embd
        elif config.norm_layer == "BatchNormIdPure":
            self.norm_size = config.block_size
        elif config.norm_layer in ["BatchNorm", "InstanceNorm"]:
            self.norm_size = config.block_size
        elif config.norm_layer == "BatchNormTranspose":
            self.norm_size = config.n_embd
        elif config.norm_layer == "BatchNormIdNoReplace":
            self.norm_size = config.block_size
        elif config.norm_layer == "None":
            self.norm_size = None
        else:
            raise AttributeError("cannot determine model for norm layer: " + config.norm_layer)
        log.debug(print(config.vocab_size, config.n_embd))
        if self.norm_size:
            ln_out = getattr(custom_nn, config.norm_layer, None)
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                emb_add = custom_nn.EltwiseAdd(),
                dropout = nn.Dropout(config.dropout),
                layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
                ln_out = ln_out(self.norm_size, config.bias),
            ))
        else:
            ln_out = nn.Identity
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd),
                emb_add = custom_nn.EltwiseAdd(),
                dropout = nn.Dropout(config.dropout),
                layer = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)]),
            ))

        self.linear_out = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate

        # https://paperswithcode.com/method/weight-tying
        if self.use_weight_tying:
            self.transformer.wte.weight = self.linear_out.weight

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        log.info("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):

        device = idx.device
        b, t = idx.size()
        #print(f'{idx}----------{idx.size()}')
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"
        pos = torch.arange(0, t, dtype=torch.long, device=device).unsqueeze(0) # shape (1, t)

        # forward the GPT model itself
        #TODO: add padding for FINN support
        tok_emb = self.transformer.wte(idx) # token embeddings of shape (b, t, n_embd)
        pos_emb = self.transformer.wpe(pos) # position embeddings of shape (1, t, n_embd)
        x = self.transformer.emb_add(tok_emb, pos_emb)
        x = self.transformer.dropout(x)
        for block in self.transformer.layer:
            x = block(x)
        if self.norm_size:
            x = self.transformer.ln_out(x)

        loss = None
        if targets is None and self.single_output:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.linear_out(x[:, [-1], :]) # note: using list [-1] to preserve the time dim
        else:
            # if we are given some desired targets also calculate the loss
            logits = self.linear_out(x)
            if targets is not None:
                #squeeze batch and block_size dimension together, retain non-softmaxed word probabilities
                #logits become a 1d tensor, containing the index of the next word
                if self.config.shift_targets:
                    # move labels to correct device to enable model parallelism
                    targets = targets.to(logits.device)
                    # Shift so that tokens < n predict n
                    shift_logits = logits[..., :-1, :].contiguous()
                    shift_labels = targets[..., 1:].contiguous()
                    # Flatten the tokens
                    from torch.nn import CrossEntropyLoss
                    loss_fct = CrossEntropyLoss()
                    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
                else:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1)

        return logits, loss

    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model, model_type):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}

        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        config_args['vocab_size'] = 50257 # always 50257 for GPT model checkpoints
        config_args['block_size'] = 1024 # always 1024 for GPT model checkpoints
        config_args['bias'] = True # always True for GPT model checkpoints
        print("our model")
        print(model)
        config = model.config 
        print(config)
        sd = model.state_dict()
        sd_keys = sd.keys()
        sd_keys = [k for k in sd_keys if not k.endswith('.attn.bias')] # discard this mask / buffer, not a param


        ### troch model import
        #import torch
        #model_t = torch.hub.load('huggingface/transformers', 'modelForCausalLM', 'gpt2')
        #model_t_sd_keys = model_t.state_dict().keys()
        #print("torch model")
        #print(model_t)
        #print(model_t_sd_keys)

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        print("theirs")
        print(model_hf)
        sd_hf = model_hf.state_dict()
        # copy while ensuring all of the parameters are aligned and match in names and shapes
        sd_keys_hf = sd_hf.keys()
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.masked_bias')] # ignore these, just a buffer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.attn.bias')] # same, just the mask (buffer)

        
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.ln_1.weight')] # ignore norm layers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.ln_1.bias')] # ignore norm layers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.ln_2.weight')] # ignore norm layers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.ln_2.bias')] # ignore norm layers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.ln_f.weight')] # ignore norm layers
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('.ln_f.bias')] # ignore norm layers

        # fixed by weight tying anyways?
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('lm_head.weight')] # ignore head layer
        sd_keys_hf = [k for k in sd_keys_hf if not k.endswith('lm_head.bias')] # ignore head layer

        transposed = ['attn.c_attn.weight', 'attn.c_proj.weight', 'mlp.c_fc.weight', 'mlp.c_proj.weight']
        # basically the openai checkpoints use a "Conv1D" module, but we only want to use a vanilla Linear
        # this means that we have to transpose these weights when we import them

        # ignore missing keys
        # assert len(sd_keys_hf) == len(sd_keys), f"mismatched keys: {len(sd_keys_hf)} != {len(sd_keys)}"

        for k in sd_keys_hf:
            # h == layer in our model
            if k.startswith("transformer.h"):   
                ak = k.split('.')
                ak[1] = 'layer'
                _k = '.'.join(ak)
            else:
                _k = k

            print(_k)
            print(sd[_k].shape)
            print(k)
            print(sd_hf[k].shape[::-1])

            if any(k.endswith(w) for w in transposed):
                # special treatment for the Conv1D weights we need to transpose
                assert sd_hf[k].shape[::-1] == sd[_k].shape
                with torch.no_grad():
                    sd[_k].copy_(sd_hf[k].t())
            else:
                # vanilla copy over the other parameters
                assert sd_hf[k].shape == sd[_k].shape
                with torch.no_grad():
                    sd[_k].copy_(sd_hf[k])
        

        return model

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx
