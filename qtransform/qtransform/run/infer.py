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
from os.path import isdir, exists, join, expanduser, isabs
from os import makedirs, getcwd, makedirs
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
from datetime import datetime
from . import generate
from . import InferType

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
    name: str = None
    encoding: str = None
    meta_path: str = None

@dataclass
class ONNXConfig():
    """
    Boilerplate class to represent infer config for ONNX models as ONNX models cannot save generic metadata about the training process
    such as dataset info, tokenizers, epochs etc. Instead, the data has to be supplied within the infer config to make sure that the generated
    tokens of the model are decoded correctly.
    """
    tokenizer: TokenizerConfig
    path: str = None

    def __setattr__(self, name, value):
        if name == "tokenizer":
            if not isinstance(value, Union[Dict, DictConfig, TokenizerConfig]):
                log.error(f'Attribute tokenizer of class ONNXConfig can only support Dicts or TokenizerConfig')
                raise TypeError()
            if isinstance(value, Union[Dict, DictConfig]):
                value = TokenizerConfig(**value)
        self.__dict__[name] = value


def load_model(cfg: DictConfig, device: torch.device) -> List[ModelData]:
    """
    Loads an ONNX model and a torch checkpoint from paths supplied in the infer config.
    The function returns a dictionary with the loaded models, ready to be used for inference.
    The dictionary contains the keys "onnx" for ONNX models and "checkpoint" for torch checkpoints.
    TODO: determine how many models/checkpoints can be loaded in one run process and if loading both models
    and checkpoints is a good idea.
    """
    #if supplied, run inference on both onnx model and checkpoint
    models: List[ModelData] = list()
    onnx_model = ONNXConfig(**cfg.run.get('onnx_model', dict()))
    from_checkpoint_path = cfg.run.get('from_checkpoint', '')
    #onnx checkpoint
    if onnx_model.path != None:
        log.critical(f'Inference for ONNX models not implemented yet')
        raise NotImplementedError()

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
        #TODO: maybe implement a generic type checking method
        compile_model = cfg.run.get('compile', False)
        if not isinstance(compile_model, bool):
            log.warning(f'compile should be set to True or False, not {compile_model}. Defaulting to: False')
            compile_model = False
        if torch.__version__ >= (2,0) and compile_model:
            model = torch.compile(model) # requires PyTorch 2.0 (optional)
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
                #inference params, start prompt written in hex and in plain character
                file.write(f'num_samples: {num_samples}, max_new_tokens: {max_new_tokens}, temperature: {temperature}, top_k: {top_k}\n'\
                    f'start prompt: {[hex(ord(x)) for x in start]} ("{start}")\n')
                file.write(f'----------- BEGIN INFERENCE -----------\n')
                for i, text in enumerate(gen_infer, start=1):
                    log.info(f'Writing sample: {i}/{num_samples}')
                    file.write(text)
                log.info(f'Finished writing into file "{out_path}".')
        else:
            for text in gen_infer:
                print(text)


