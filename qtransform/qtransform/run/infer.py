import logging
log = logging. getLogger(__name__)
from omegaconf import DictConfig
from torch import nn
import torch

def run(cfg : DictConfig):
    """ Inference """
    log.info("=================")
    log.info("Running Inference")
    log.info("=================")
    return
    cuda = None
    device = None
    if "cuda" in cfg:
        cuda = cfg.cuda and torch.cuda.is_available()
    else:
        cuda = torch.cuda.is_available()
    mps = None
    if "mps" in cfg:
        mps = cfg.mps and torch.backends.mps.is_available()
    else:
        mps = torch.backends.mps.is_available()

    torch.manual_seed(cfg.seed)    
    if cuda:
        device = torch.device("cuda")
        cuda_kwargs = {'pin_memory': True,}
        cfg.dataset.dataloader.update(cuda_kwargs)
    elif mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"using device: {str(device)}")
    log.info(f"number of torch dataloader: {str(cfg.dataset.dataloader.num_workers)}")

    from qtransform.model import get_model
    model = get_model(cfg.model)
    model.eval()
    model.to(device)

    return infer(cfg, model)


def infer(cfg: DictConfig, model: nn.Module):
    """
    Sample from a trained model
    """
    import os
    import pickle
    import torch
    import tiktoken

    # -----------------------------------------------------------------------------
    start = "\n" # or "<|endoftext|>" or etc. Can also specify a file, use as: "FILE:prompt.txt"
    num_samples = 10 # number of samples to draw
    max_new_tokens = 500 # number of tokens generated in each sample
    temperature = 0.8 # 1.0 = no change, < 1.0 = less random, > 1.0 = more random, in predictions
    top_k = 200 # retain only the top_k most likely tokens, clamp others to have 0 probability
    compile = False # use PyTorch 2.0 to compile the model to be faster
    exec(open('configurator.py').read()) # overrides from command line or config file
    # -----------------------------------------------------------------------------

    # model
    if init_from == 'resume':
        # init from a model saved in a specific directory
        ckpt_path = os.path.join(out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device)
        gptconf = GPTConfig(**checkpoint['model_args'])
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
    elif init_from.startswith('gpt2'):
        # init from a given GPT-2 model
        model = GPT.from_pretrained(init_from, dict(dropout=0.0))

    model.eval()
    model.to(device)
    if compile:
        model = torch.compile(model) # requires PyTorch 2.0 (optional)

    # look for the meta pickle in case it is available in the dataset folder
    load_meta = False
    if init_from == 'resume' and 'config' in checkpoint and 'dataset' in checkpoint['config']: # older checkpoints might not have these...
        meta_path = os.path.join('data', checkpoint['config']['dataset'], 'meta.pkl')
        load_meta = os.path.exists(meta_path)
    if load_meta:
        print(f"Loading meta from {meta_path}...")
        with open(meta_path, 'rb') as f:
            meta = pickle.load(f)
        # TODO want to make this more general to arbitrary encoder/decoder schemes
        stoi, itos = meta['stoi'], meta['itos']
        encode = lambda s: [stoi[c] for c in s]
        decode = lambda l: ''.join([itos[i] for i in l])
    else:
        # ok let's assume gpt-2 encodings by default
        print("No meta.pkl found, assuming GPT-2 encodings...")
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
    with torch.no_grad():
        with ctx:
            for k in range(num_samples):
                y = model.generate(x, max_new_tokens, temperature=temperature, top_k=top_k)
                print(decode(y[0].tolist()))
                print('---------------')
