import logging
from typing import Any

from qtransform import device_singleton
log = logging. getLogger(__name__)
from omegaconf import DictConfig, open_dict
from torch import nn
import torch
import tiktoken
from torch.nn import functional as F
from qtransform import device_singleton
from dataclasses import dataclass

@dataclass
class InferConfig():
    command: str =  "infer"

    start: str = "\n"
    model_dir: str = "models"
    from_checkpoint: str = None #filename of checkpoint to load

    num_samples: int = 10 #generate num_samples 
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200

    to_file: str = None

    onnx_model: str = None

def run(cfg : DictConfig):
    """ Inference """
    log.info("=================")
    log.info("Running Inference")
    log.info("=================")
    
    device_singleton.device = cfg.device
    device = device_singleton.device

    torch.manual_seed(cfg.seed)    
    if device.type == "cuda":
        cuda_kwargs = {'pin_memory': True,}
        #struct flag of dictconf prevents additional keys to be added (https://omegaconf.readthedocs.io/en/latest/usage.html#struct-flag)
        with open_dict(cfg.dataset.dataloader):
            cfg.dataset.dataloader.update(cuda_kwargs)
    log.info(f"using device: {str(device)}")
    infer(cfg, device)

def infer(cfg: DictConfig, device: Any):
    """
    Sample from a trained model. It prints the predicted words onto stdout
    """
    # -----------------------------------------------------------------------------
    infer_cfg: InferConfig = InferConfig(**cfg.run)
    start = infer_cfg.start
    num_samples = infer_cfg.num_samples # number of samples to draw
    max_new_tokens = infer_cfg.max_new_tokens # number of tokens generated in each sample
    temperature = infer_cfg.temperature # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = infer_cfg.top_k # retain only the top_k most likely tokens, clamp others to have 0 probability
    # -----------------------------------------------------------------------------

    #load model from checkpoint
    from qtransform.utils import load_checkpoint
    epoch, checkpoint = load_checkpoint(cfg=cfg)
    model_cfg = checkpoint.get('model_cfg')
    if model_cfg is None:
        log.warning(f'No model config in checkpoint specified. Inferring from hydra config.')
        model_cfg = cfg.get("model")
    if model_cfg is None:
        log.error(f'No model config specified.')
        raise KeyError()

    from qtransform.model import get_model
    model = get_model(model_cfg)
    model.eval()
    model.to(device)

    if torch.__version__ >= (2,0):
        model = torch.compile(model) # requires PyTorch 2.0 (optional)
    # load tokenizer to decode tokens properly
    # tokenizer info saved in checkpoint or in hydra config
    from qtransform.dataset.tokenizer import get_tokenizer, Tokenizer
    tokenizer_cfg = checkpoint.get("tokenizer_cfg")
    if tokenizer_cfg is None:
        log.warning(f'Model checkpoint does not contain tokenizer information. Using tokenizer info from config')
        tokenizer_cfg = cfg.dataset.get("tokenizer")
    if tokenizer_cfg is None:
        log.error(f'Tokenizer configuration neither specified in model checkpoint nor in hydra config.')
        raise KeyError()
    tokenizer: Tokenizer = get_tokenizer(tokenizer_cfg)
    encode = tokenizer.encode
    decode = tokenizer.decode

    #load metadata, including vocabulary for character tokenization
    log.debug(checkpoint["tokenizer_cfg"]["meta"])
    tokenizer.load_metadata(meta=checkpoint["tokenizer_cfg"]["meta"])
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
        # the results should not be softmaxed yet as they will be later within this function
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