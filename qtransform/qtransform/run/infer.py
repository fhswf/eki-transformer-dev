import logging
from typing import Any

from qtransform import device_singleton
log = logging. getLogger(__name__)
from omegaconf import DictConfig
from torch import nn
import torch
import tiktoken
from torch import functional as F

def run(cfg : DictConfig):
    """ Inference """
    log.info("=================")
    log.info("Running Inference")
    log.info("=================")
    
    torch.manual_seed(cfg.seed)    
    device_singleton.device = cfg.device
    device = device_singleton.device

    from qtransform.model import get_model
    model = get_model(cfg.model)
    model.eval()
    model.to(device)

    return infer(cfg, model, device)


def infer(cfg: DictConfig, model: nn.Module, device: Any):
    """
    Sample from a trained model
    """
    # -----------------------------------------------------------------------------
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 10 # number of samples to draw
    max_new_tokens = 500 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    # -----------------------------------------------------------------------------

    from qtransform.utils import load_checkpoint
    from_epoch, checkpoint = load_checkpoint(cfg=cfg)
    if torch.__version__ >= 2.0:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # ok let's assume gpt-2 encodings by default
    log.warn("assuming GPT-2 encodings...")
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda l: enc.decode(l)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
    start_ids = encode(start)
    x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])

    # run generation
    for k in range(num_samples):
        y = generate(model, x, max_new_tokens, temperature=temperature, top_k=top_k)
        print(decode(y[0].tolist()))
        print('---------------')

@torch.no_grad()
def generate(model, idx, max_new_tokens, temperature=1.0, top_k=None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size:]
        # forward the model to get the logits for the index in the sequence
        logits, _ = model(idx_cond)
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