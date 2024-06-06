from typing import Tuple, Union, List, Dict, Any
from omegaconf import DictConfig, open_dict
import os
from dataclasses import dataclass
from qtransform.tokenizer.tokenizer import Tokenizer
from qtransform.utils.helper import load_checkpoint, load_onnx_model
from qtransform.utils.introspection import get_classes
from qtransform.model import QTRModelWrapper, ModelType
import torch
from torch import nn
import torch.nn.functional as F
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
from enum import Enum
from logging import getLogger

log = getLogger(__name__)

def prepare_run(cfg: DictConfig): 
    """
    all variables below could be singletons
        model (could also be singleton?)
        dataloaderwrapper (if needed)
        optimizer (if needed)
        scheduler (if needed)
        set tokenizer singleton
    """
    pass


@torch.no_grad()
def generate(model_wrapper: QTRModelWrapper, idx: torch.Tensor, max_new_tokens: int, temperature: float =1.0, top_k: int =None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    This works for both ONNX models and torch Modules.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    if not isinstance(model_wrapper, QTRModelWrapper):
        log.error(f'Cannot generate text without QTRModelWrapper instance')
        raise TypeError()
    if model_wrapper.model_type == ModelType.PRETRAINED:
        log.warning(f'Inference for pretrained models not tested yet')
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        block_size = model_wrapper.model_cfg.args.block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        # the results should not be softmaxed yet as they will be later within this function
        logits, _ = model_wrapper(idx_cond)
        # pluck the logits at the final step and scale by desired temperature
        if temperature is not None and temperature > 0 and  temperature > 1.0e-10:
            logits = logits[:, -1, :] / temperature
        else:
            logits = logits[:, -1, :] 
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

