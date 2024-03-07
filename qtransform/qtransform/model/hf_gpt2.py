from transformers import GPT2LMHeadModel, GPT2Config
from typing import Dict, Union
from omegaconf import DictConfig
from torch.nn import Module

class HuggingfaceGPT2LMHeadModel(GPT2LMHeadModel):
    """
        Rename class of Hugginface's GPT2 model to be a bit more verbose.
        This class serves for debugging purposes to compare huggingface's implementation of GPT2
        to our adjusted version of karpathy's GPT2 model.
        Huggingface GPT cannot change the normalization layer
    """
    def __init__(self, config: Union[Dict, DictConfig, GPT2Config]):
        if isinstance(config, Dict):
            config = DictConfig(**config)
        if isinstance(config, DictConfig):
            #huggingface config uses other config names to ours
            dropout = config.dropout if config.dropout is not None else 0.0
            config = dict(
                n_layer = config.n_layer,
                n_head = config.n_head,
                n_embd = config.n_embd,
                n_positions = config.block_size,
                n_inner = config.n_embd * 4,
                embd_pdrop  = dropout,
                attn_pdrop  = dropout,
                resid_pdrop = dropout,
                activation_function = config.transformer_active_func.lower()
                )
            config = GPT2Config(**config)
        super().__init__(config)
    def forward(self, input_ids, labels = None):
        #attention mask not necessary as our tokens are not padded
        out = super().forward(input_ids = input_ids, labels = labels) 
        #loss can be None if no labels are supplied
        return out.logits, out.loss

PRETRAINED_VERSIONS = ["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"]
class PreTrainedGPT2(Module):
    """
    Wrapper around pretrained GPT2 model to make our workflow of finetuning and running inference/benchmarking compatible
    """
    def __init__(self, config: DictConfig):
        version = config.version
        if not isinstance(version, str) or version not in PRETRAINED_VERSIONS:
            log.error(f'Pretrained model should be one of: {PRETRAINED_VERSIONS}, not: {cfg.version}')
            raise ValueError()
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(version)

    def forward(self, inputs, labels = None):
        #rint(inputs)
        #rint(labels)
        if labels is not None:
            labels = inputs
        #print(inputs)
        #print(labels)
        #print(labels[..., 1:].contiguous())
        out = self.model(inputs, labels=labels)
        #no loss, could be implemented here or in train.py
        return out.logits, out.loss
       