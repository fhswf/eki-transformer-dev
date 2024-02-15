from typing import Union
import torch
from torch import nn
import torch.nn.functional as F
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
from enum import Enum

class InferType(Enum):
    ONNX = 1
    CHECKPOINT = 2


@torch.no_grad()
def generate(model_type: InferType, model: Union[nn.Module, ModelWrapper], idx: torch.Tensor, max_new_tokens: int, temperature: float =1.0, top_k: int =None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    This works for both ONNX models and torch Modules.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """

    def forward_pass(idx_cond: torch.Tensor) -> torch.Tensor:
        #generic function wrapper as forward pass for onnx models and torch modules is different
        logits = None
        match model_type:
            case InferType.ONNX:
                idict = {"input": idx.numpy()}
                # use infer_shapes()
                #forward pass of gpt model returns the non-softmaxed token predictions
                odict = execute_onnx(model, idict)
                logits = torch.from_numpy(odict["output"])
            case InferType.CHECKPOINT:
                logits, _ = model(idx_cond)
            case _:
                log.error(f'Forward pass only supported for ONNX models or checkpoints')
                raise ValueError()
        return logits

    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        # the results should not be softmaxed yet as they will be later within this function
        logits = forward_pass(idx_cond)
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