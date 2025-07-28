import logging
from typing import Any, Dict, Generator, Union, List
from dataclasses import dataclass
from omegaconf import DictConfig, open_dict
import hydra
from omegaconf import OmegaConf
from torch import nn
import torch
import tiktoken
from qtransform import device_singleton
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from qtransform.tokenizer.tokenizer import Tokenizer
from qtransform.model import ModelType, get_model_wrapper, QTRModelWrapper,DynamicCheckpointQTRModelWrapper
from os.path import isdir, exists, join, expanduser, isabs
from os import makedirs, getcwd, makedirs
from datetime import datetime
import numpy as np
from functools import lru_cache
import torch.nn.functional as F

# Keep track of 10 different messages and then warn again
@lru_cache(1)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)

log = logging.getLogger(__name__)

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
    infer(cfg, device=device_singleton.device)
    
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
    
    block_size = model_wrapper.model_cfg.args.block_size
    # get block size from onnx model if it is an onnx model
    if block_size is None:
        if model_wrapper.model_type == ModelType.ONNX:
            shape = []
            tensor_type = model_wrapper.model.graph.input[0].type.tensor_type
            for dim in tensor_type.shape.dim:
                # dim.dim_value is 0 if dynamic, otherwise the static value
                shape.append(dim.dim_value)
            block_size = shape[1] if len(shape) > 1 else None
    log.info(f'Using block_size: {block_size} for model {model_wrapper.model_type.name}')
    
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
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


#TODO: huggingface makes use of pipelines for inference
#(https://huggingface.co/docs/transformers/main_classes/pipelines)
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
    #model_wrapper.model.to(device=device)
    
    quant_cfg = cfg.get('quantization')
    if quant_cfg and quant_cfg.quantize:
        if not model_wrapper.quantized:    
            log.info(f'Quantizing model')
            model_wrapper.quantize_model(quant_cfg)
        else:
            warn_once(log, f'Model was already quantized, ignoring quant_cfg from hydra')
        #from qtransform.quantization import get_quantizer
        #quantizer, model_quant_cfg = get_quantizer(quant_cfg, model=model)
        #model, replace_layers_later = quantizer.get_quantized_model(model_quant_cfg, inplace=True)
        #quantize last layers (batchnorm). parmams last saved checkpoint do not entirely reflect current model anymore 
        #if replace_layers_later is not None:
        #    model, _ = quantizer.get_quantized_model(replace_layers_later)
        assert isinstance(model_wrapper, DynamicCheckpointQTRModelWrapper), f'Model should be torch module, not {type(model_wrapper)}'
    #only parameters (type torch.nn.parameter.Parameter) are moved to the device, not non-named Tensors
    #this is a problem if a layer uses a non-named Tensor during the forward pass
    model_wrapper.to(device=device_singleton.device)
    
    # try and get tokenizer conf from checkpoint
    # TODO do we have a generic way via the picke file? To combine model and tokenizer
    # TODO actually Tokenizer is very much bound to the model for all steps: train, test, infer. Why not combine them completely in QTRModelWrapper?)
    if OmegaConf.is_missing(cfg, "tokenizer") or OmegaConf.is_missing(cfg, "tokenizer.encodig") or cfg.get("tokenizer.encodig") is None:
        if hasattr(model_wrapper, "tokenizer_cfg"):
            OmegaConf.update(cfg, "tokenizer", model_wrapper.tokenizer_cfg)
    
    # encode the beginning of the prompt
    if start.startswith('FILE:'):
        with open(start[5:], 'r', encoding='utf-8') as f:
            start = f.read()
            start = start[:-1]
            print(start)
            
    def write_inference(model_wrapper: QTRModelWrapper) -> Generator[str, None, None]:
        """
        Runs inference on the models, yielding the generated text.
        The implementation of this method is sort of dirty as depending on the type of model and the type of tokenizer,
        the start prompt has to be tokenized and passed differently. However, some params such as the number of tokens, temperature
        etc. are not passed as args.
        """
        tokenizer_singleton.tokenizer = cfg.tokenizer
        tokenizer = tokenizer_singleton.tokenizer
        print(f"{type(tokenizer)=}")
        # TODO is this warning still up to date?
        log.warning(f'Dataset and tokenizer usage is still a WIP, for now gpt2 tiktokenizer is used for inference')
        start_ids = tokenizer.encode(start)
        x = (torch.tensor(start_ids, dtype=torch.long, device=device)[None, ...])
        log.info(f'Running inference from {model_wrapper.model_type.name}.')
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