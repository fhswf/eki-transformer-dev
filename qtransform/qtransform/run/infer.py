import logging
from typing import Any, Dict, Union, List

from qtransform import device_singleton
log = logging. getLogger(__name__)
from omegaconf import DictConfig, open_dict
from torch import nn
import torch
import tiktoken
from torch.nn import functional as F
from qtransform import device_singleton
from qtransform.utils import load_checkpoint, load_onnx_model
from qtransform.dataset.tokenizer import get_tokenizer, Tokenizer
from dataclasses import dataclass
from os.path import isdir, exists, join, expanduser

from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
from datetime import datetime

@dataclass
class InferConfig():
    command: str =  "infer"

    start: str = "\n"
    model_dir: str = "models"
    from_checkpoint: str = None #filename of checkpoint to load
    onnx_model: str = None

    num_samples: int = 10 #generate num_samples 
    max_new_tokens: int = 500
    temperature: float = 0.8
    top_k: int = 200

    out_dir: str = None

    onnx_model: dict = None

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

from enum import Enum

class InferType(Enum):
    ONNX = 1
    CHECKPOINT = 2

@dataclass
class ModelData():
    """
    Dataclass to store the saved model, the type (onnx model or torch module) and the corresponding
    tokenizer for that model.
    """
    type: InferType 
    model: Union[nn.Module, ModelWrapper]
    tokenizer: Tokenizer

@dataclass
class TokenizerConfig():
    name: str = ""
    encoding: str = ""
    meta_path: str = ""

@dataclass
class ONNXConfig():
    """
    Boilerplate class to represent infer config for ONNX models as ONNX models cannot save generic metadata about the training process
    such as dataset info, tokenizers, epochs etc. Instead, the data has to be supplied within the infer config to make sure that the generated
    tokens of the model are decoded correctly.
    """
    path: str = ""
    tokenizer: TokenizerConfig

    def __setattr__(self, name, value):
        if name == "tokenizer":
            if not isinstance(value, Union[Dict, DictConfig, TokenizerConfig]):
                log.error(f'Attribute tokenizer of class ONNXConfig can only support Dicts or TokenizerConfig')
                raise TypeError()
            if isinstance(value, Union[Dict, DictConfig]):
                tokenizer = TokenizerConfig(**value)
            self.tokenizer = tokenizer


def load_model(cfg: DictConfig, device: torch.device) -> List[ModelData]:
    """
    Loads an ONNX model and a torch checkpoint from paths supplied in the infer config.
    The function returns a dictionary with the loaded models, ready to be used for inference.
    The dictionary contains the keys "onnx" for ONNX models and "checkpoint" for torch checkpoints.
    """
    #if supplied, run inference on both onnx model and checkpoint
    models: List[ModelData] = list()
    onnx_model = ONNXConfig(cfg.run.get('onnx_model', dict()))
    from_checkpoint_path = cfg.run.get('from_checkpoint', '')
    #onnx checkpoint
    if onnx_model["path"] != None:
        model = load_onnx_model(onnx_model["path"])
        #TODO: load tokenizer cfg from infer config
        from qtransform.dataset.tokenizer import __all__

        models.append(ModelData(type=InferType.ONNX, model=model, tokenizer=tokenizer))
    #torch checkpoint
    if from_checkpoint_path != None:
        #load model from checkpoint
        epoch, checkpoint = load_checkpoint(cfg=cfg)
        model_cfg = checkpoint.get('model_cfg')
        if model_cfg is None:
            log.warning(f'No model config in checkpoint specified. Inferring from hydra config.')
            model_cfg = cfg.get("model")
        if model_cfg is None:
            log.error(f'No model config specified.')
            raise KeyError()
        if "quantized" not in checkpoint:
            log.warning(f'No info specified if checkpoint is quantized. Assuming false.')
        elif checkpoint["quantized"]:
            log.warning(f'Behavior might be unexpected as checkpoint possibly contains quantized params.')
        from qtransform.model import get_model
        model = get_model(model_cfg)
        model.eval()
        model.to(device)
        #if torch.__version__ >= (2,0):
        #    model = torch.compile(model) # requires PyTorch 2.0 (optional)
        #tokenizer for model
        # tokenizer info saved in checkpoint or in hydra config
        tokenizer_cfg = checkpoint.get("tokenizer_cfg")
        if tokenizer_cfg is None:
            log.warning(f'Model checkpoint does not contain tokenizer information. Using tokenizer info from config')
            tokenizer_cfg = cfg.dataset.get("tokenizer")
        if tokenizer_cfg is None:
            log.error(f'Tokenizer configuration neither specified in model checkpoint nor in hydra config.')
            raise KeyError()
        tokenizer: Tokenizer = get_tokenizer(tokenizer_cfg)
        #load metadata, including vocabulary for character tokenization
        log.debug(tokenizer_cfg["meta"])
        tokenizer.load_metadata(meta=tokenizer_cfg["meta"])

        models.append(ModelData(type=InferType.CHECKPOINT, model=model, tokenizer=tokenizer))
    else:
        log.warning(f'Path to checkpoint "{from_checkpoint_path}" is not a file.')
    if len(models) == 0:
        log.error(f'Could not load models with fields "onnx_model": {onnx_model_path}, "from_checkpoint": {from_checkpoint_path}')
        raise ValueError()
    
    return models

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
            y = generate(model_type, model, x, max_new_tokens, temperature=temperature, top_k=top_k)
            log.debug(f'Uniquely generated tokens, sorted in ascending order: {y.unique().sort()}')
            #TODO: model could have larger vocabulary size than the tokenizer's max_token_value
            #      for character tokenization, a sequence of <UNKNOWN> chars will be printed. for tiktoken, inference crashes
            yield tokenizer.decode(y[0].tolist()) + '\n---------------\n'

    out_dir = cfg.run.get('out_dir', '')
    #infer for onnx and checkpoint
    for model_data in models:
        #inference yields generator in case something should be done before returning entire output
        gen_infer = write_inference(model_data)
        #write samples into file
        if out_dir is not None and isdir(out_dir):
            timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
            filename = "INFER_" + timestamp + "_" + model_data.type.name + ".out"
            out_path = join(out_dir.replace('~', expanduser('~')), filename)
            with open(out_path, 'w') as file:
                log.info(f'Writing to file: "{out_path}"')
                #inference params, start prompt written in hex and in plain character
                file.write(f'num_samples: {num_samples}, max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}\n'\
                    f'start prompt: {[hex(ord(x)) for x in start]} ("{start}")\n')
                file.write(f'----------- BEGIN INFERENCE -----------\n')
                for i, text in enumerate(gen_infer):
                    log.info(f'Writing sample: {i}/{num_samples}')
                    file.write(text)
                log.info(f'Finished writing into file "{out_path}".')
        else:
            for text in gen_infer:
                print(text)


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