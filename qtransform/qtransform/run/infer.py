import logging
from typing import Any, Dict, Union, List
from dataclasses import dataclass
from qtransform import device_singleton
log = logging.getLogger(__name__)
from omegaconf import DictConfig, open_dict
from torch import nn
import torch
import tiktoken
from qtransform import device_singleton
from os.path import isdir, exists, join, expanduser, isabs
from os import makedirs, getcwd, makedirs
from datetime import datetime
from . import generate, load_model, ModelData
import matplotlib.pyplot as plt
import numpy as np

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

def run(cfg : DictConfig):
    """ Inference """
    log.info("=================")
    log.info("Running Inference")
    log.info("=================")
    #cuda does not work as some unnamed tensors are still on cpu
    #TODO: find them and make them parameters
    #device_singleton.device = cfg.device
    device_singleton.device = 'cpu'
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

    models: ModelData = load_model(cfg, device)

    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()

    def write_inference(model_data: ModelData) -> str:
        """
        Runs inference on the models, yielding the generated text.
        The implementation of this method is sort of dirty as depending on the type of model and the type of tokenizer,
        the start prompt has to be tokenized and passed differently. However, some params such as the number of tokens, temperature
        etc. are not passed as args.
        """
        model_type = model_data.type
        model = model_data.model
        tokenizer = model_data.tokenizer
        #TODO: infer vocab size of onnx model
        #max_token_value = tokenizer.meta.max_token_value
        #if max_token_value < model_cfg.args.vocab_size:
        #    log.warning(f'Vocab size of model is larger than the tokenizer vocab. '\
        #        'This could lead to errors when the model predicts a token id that is not present within the vocab.')
        #tokens can be different if onnx model and model from checkpoint have different tokenizers
        #not sure when that use case would be necessary though
        start_ids = tokenizer.encode(start, infer=True)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        log.info(f'Running inference from {model_type.name.upper()}.')
        for k in range(num_samples):
            y: torch.Tensor = generate(model_type, model, x, max_new_tokens, temperature=temperature, top_k=top_k)
            #i assume that sorting will take a long time which is redundant without debugging purposes
            if cfg.debug:
                #log.debug(f'Uniquely generated tokens, sorted in ascending order: {y.unique().sort().values}')
                log.debug(f'Highest predicted token: {torch.max(y)}')
            #TODO: catch Panic Exception in case token ids do not appear in tokenizer vocab
            yield tokenizer.decode(y[0].tolist()) + '\n---------------\n'

    out_dir = cfg.run.get('out_dir', '')
    #infer for onnx and checkpoint
    for model_data in models:
        #inference yields generator in case something should be done before returning entire output
        gen_infer = write_inference(model_data)
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
            filename = "INFER_" + timestamp + "_" + model_data.type.name + ".out"
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


