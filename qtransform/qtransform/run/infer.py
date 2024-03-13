import logging
from typing import Any, Dict, Union, List
from dataclasses import dataclass
from qtransform import device_singleton
log = logging.getLogger(__name__)
from omegaconf import DictConfig, open_dict
import hydra
from torch import nn
import torch
import tiktoken
from qtransform import device_singleton
from os.path import isdir, exists, join, expanduser, isabs
from os import makedirs, getcwd, makedirs
from datetime import datetime
from . import generate
from qtransform.model import get_model_wrapper, QTRModelWrapper
import numpy as np
from qtransform.dataset.tokenizer.tokenizer import Tokenizer

@dataclass
class InferConfig():
    command: str =  "infer"

    start: str = "\n"
    checkpoint_dir: str = "models"
    from_checkpoint: str = None #filename of checkpoint to load
    onnx_model: str = None

    num_samples: int = 10 #generate num_samples 
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200

    out_dir: str = None

    onnx_model: dict = None
    compile: bool = True

    debug: bool = False

    pretrained_model: str = None

def run(cfg : DictConfig):
    """ Inference """
    log.info("=================")
    log.info("Running Inference")
    log.info("=================")
    device_singleton.device = cfg.device
    device = device_singleton.device

    torch.manual_seed(cfg.seed)
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
    out_dir = infer_cfg.out_dir
    # -----------------------------------------------------------------------------
    #TODO: make inference work for huggingface pretrained models
    model_wrapper: QTRModelWrapper = get_model_wrapper(cfg.model)
    model_wrapper.model.to(device=device)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()

    def write_inference(model_wrapper: QTRModelWrapper) -> str:
        """
        Runs inference on the models, yielding the generated text.
        The implementation of this method is sort of dirty as depending on the type of model and the type of tokenizer,
        the start prompt has to be tokenized and passed differently. However, some params such as the number of tokens, temperature
        etc. are not passed as args.
        """
        tokenizer = tiktoken.get_encoding("gpt2") #TODO get tokenizer
        log.warning(f'Dataset and tokenizer usage is still a WIP, for now gpt2 tiktokenizer is used for inference')
        start_ids = tokenizer.encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        log.info(f'Running inference from {model_wrapper.model_type}.')
        for k in range(num_samples):
            #TODO: block_size for onnx models
            y: torch.Tensor = generate(model_wrapper = model_wrapper, idx = x, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k)
            #i assume that sorting will take a long time which is redundant without debugging purposes
            if cfg.debug:
                #log.debug(f'Uniquely generated tokens, sorted in ascending order: {y.unique().sort().values}')
                log.debug(f'Highest predicted token: {torch.max(y)}')
            #TODO: catch Panic Exception in case token ids do not appear in tokenizer vocab
            yield tokenizer.decode(y[0].tolist()) + '\n---------------\n'

    out_dir = cfg.run.get('out_dir', '')
    if isinstance(model_wrapper.model, nn.Module):
        if torch.__version__ >= (2,0) and cfg.run.compile: 
            model_wrapper.model = torch.compile(model_wrapper.model) # requires PyTorch 2.0 (optional)
        model_wrapper.model.eval()
    gen_infer = write_inference(model_wrapper)
    #write samples into file
    if out_dir is not None and len(out_dir) > 0:
        if not isabs(out_dir):
            try:
                out_path = join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, out_dir)
            except:
                out_path = join(getcwd(), out_dir)
        out_path = out_path.replace('~', expanduser('~'))
        if not exists(out_path):
            log.debug(f'Creating infer dir: {out_path}')
            makedirs(out_path, exist_ok= True )
        timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
        filename = "INFER_" + timestamp + "_" + model_wrapper.model_type + ".out"
        out_path = join(out_path, filename)
        with open(out_path, 'w') as file:
            log.info(f'Writing to file: "{out_path}"')
            #check if highest token is within tokenizer vocab
            file.write(f'num_samples: {num_samples}, max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}\n'\
                f'start prompt: {[hex(ord(x)) for x in start]} ("{start}")\n')
            file.write(f'----------- BEGIN INFERENCE -----------\n')
            for i, text in enumerate(gen_infer, start=1):
                log.info(f'Generating sample: {i}/{num_samples}')
                file.write(text)
            log.info(f'Finished writing into file "{out_path}".')
    else:
        for i, text in enumerate(gen_infer, start=1):
            log.info(f'Generating sample: {i}/{num_samples}')
            print(text)