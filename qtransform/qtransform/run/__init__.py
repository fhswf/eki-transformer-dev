from typing import Tuple, Union, List, Dict, Any
from omegaconf import DictConfig, open_dict
import os
from dataclasses import dataclass
from qtransform.dataset.tokenizer.tokenizer import Tokenizer
from qtransform.dataset import tokenizer as tokenizer_module
from qtransform.utils.helper import load_checkpoint, load_onnx_model
from qtransform.utils.introspection import get_classes
import torch
from torch import nn
import torch.nn.functional as F
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
from enum import Enum
from logging import getLogger

log = getLogger(__name__)

class InferType(Enum):
    ONNX = 0
    CHECKPOINT = 1

@dataclass
class ModelData():
    """
    Dataclass to store the saved model, the type (onnx model or torch module) and the corresponding
    tokenizer for that model.
    """
    type: InferType 
    model: Union[nn.Module, ModelWrapper]
    #tokenizer: Tokenizer
    name: str
    #block_size: int
    #TODO: metadata (block_size)

@dataclass
class TokenizerConfig():
    """
    Config for ONNX models to specify with which tokenizer the input prompt should be encoded. It reflects the yaml config structure.
    """
    module: str = None
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
    #TODO: this is only used in inference or benchmarking, unify loading of checkpoints and onnx models for all scripts
    """
    Loads an ONNX model and a torch checkpoint from paths supplied in the infer config.
    The function returns a dictionary with the loaded models, ready to be used for inference.
    The dictionary contains the keys "onnx" for ONNX models and "checkpoint" for torch checkpoints.
    It is not advised to specify both onnx models and checkpoints due to memory usage.
    """
    #if supplied, run inference on both onnx model and checkpoint
    models: List[ModelData] = list()
    onnx_model = ONNXConfig(**cfg.run.get('onnx_model', dict()))
    from_checkpoint_path = cfg.run.get('from_checkpoint', '')
    if from_checkpoint_path != None and onnx_model.path != None:
        log.warning(f'Specified both onnx models and checkpoints to load. Expect high memory consumption.')
    #onnx checkpoint
    #TODO: test this for non-quantized models
    if onnx_model.path != None:
        model = load_onnx_model(onnx_model["path"])
        tokenizer_classes: Dict[str, Any] = get_classes(tokenizer, Tokenizer)
        tokenizer = None
        for tokenizer_cls in tokenizer_classes.values():
            if tokenizer_cls.__module__.split('.')[-1] == onnx_model.tokenizer.module:
                tokenizer = tokenizer_cls({"encoding": onnx_model.tokenizer.encoding})
        if tokenizer is None:
            log.error(f'No tokenizer found for module: {onnx_model.tokenizer.module}')
            raise ValueError()
        tokenizer: Tokenizer = tokenizer_cls({"encoding": onnx_model.tokenizer.encoding})
        tokenizer.load_metadata(onnx_model.tokenizer.meta_path)
        raise NotImplementedError()
        #TODO: retrieve context length
        models.append(ModelData(type=InferType.ONNX, model=model, tokenizer=tokenizer, name= onnx_model.path))

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
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        #TODO: maybe implement a generic type checking method
        compile_model = cfg.run.get('compile', False)
        if not isinstance(compile_model, bool):
            log.warning(f'compile should be set to True or False, not {compile_model}. Defaulting to: False')
            compile_model = False
        # does not work for export atm because of fx trace and jit compile
        #if torch.__version__ >= (2,0) and compile_model:
        #    model = torch.compile(model) # requires PyTorch 2.0 (optional)
            

        # TODO reafactor no tokinzer loading in load model
        #tokenizer for model
        # tokenizer info saved in checkpoint or in hydra config
            

        # tokenizer_cfg = checkpoint.get("tokenizer_cfg")
        # if tokenizer_cfg is None:
        #     log.warning(f'Model checkpoint does not contain tokenizer information. Using tokenizer info from config')
        #     tokenizer_cfg = cfg.dataset.get("tokenizer")
        # if tokenizer_cfg is None:
        #     log.error(f'Tokenizer configuration neither specified in model checkpoint nor in hydra config.')
        #     raise KeyError()
        # tokenizer: Tokenizer = tokenizer_module.get_tokenizer(tokenizer_cfg)
        # #load metadata, including vocabulary for character tokenization
        # log.debug(tokenizer_cfg["meta"])
        # tokenizer.load_metadata(meta=tokenizer_cfg["meta"])
        # block_size = checkpoint["model_args"]["args"]["block_size"]
        
        models.append(ModelData(type=InferType.CHECKPOINT, model=model, name = from_checkpoint_path))
    else:
        log.warning(f'Path to checkpoint "{from_checkpoint_path}" is not a file.')
    if len(models) == 0:
        log.error(f'Could not load models with fields "onnx_model": {onnx_model.path}, "from_checkpoint": {from_checkpoint_path}')
        raise ValueError()
    
    return models

def forward_pass(model_type: InferType, model: Union[nn.Module, ModelWrapper], idx_cond: torch.Tensor, labels = None) -> torch.Tensor:
    """
    Generic forward pass, abstracted for torch Modules and ONNX checkpoints. Unlike the generate function, forward_pass
    returns the non-softmaxed logits and does not append the highest predicted token into a sequence for a tokenizer to decode.
    forward_pass should be useful for measuring accuracy.
    """
    #generic function wrapper as forward pass for onnx models and torch modules is different
    ret = None
    match model_type:
        case InferType.ONNX:
            idict = {"input": idx_cond.numpy()}
            # use infer_shapes()
            #forward pass of gpt model returns the non-softmaxed token predictions
            if labels is not None:
                log.warning("labels are givin to external forwards pass wrapper but they are ignored atm for onnx runs")
            odict = execute_onnx(model, idict)
            ret = torch.from_numpy(odict["output"])
        case InferType.CHECKPOINT:
            model.eval()
            if labels is not None:
                ret = model(idx_cond, labels)
            else:
                ret = model(idx_cond)
        case _:
            log.error(f'Forward pass only supported for ONNX models or checkpoints')
            raise ValueError()
    return ret


@torch.no_grad()
def generate(model_type: InferType, block_size: int, model: Union[nn.Module, ModelWrapper], idx: torch.Tensor, max_new_tokens: int, temperature: float =1.0, top_k: int =None):
    """
    Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
    the sequence max_new_tokens times, feeding the predictions back into the model each time.
    This works for both ONNX models and torch Modules.
    Most likely you'll want to make sure to be in model.eval() mode of operation for this.
    """
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        #TODO: make this work for onnx
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        # the results should not be softmaxed yet as they will be later within this function
        logits = forward_pass(model_type, model, idx_cond)
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


from torch.utils.data import DataLoader
# block_size = model.config.block_size
def get_dataloader_and_tokenizer(cfg, block_size) -> Tuple[DataLoader]:
    """ note that the tokenize ris inside the dataloader wrapper  ...  for now """

    torch.manual_seed(cfg.seed)    
    log.info(f"number of torch dataloader: {str(cfg.dataset.dataloader.num_workers)}")

    from qtransform.dataset import get_data, get_loader, DatasetWrapper, get_dataset_wrapper, OldDatasetWrapper
    data_wrapper: DatasetWrapper = get_dataset_wrapper(cfg.dataset)
    data_wrapper.load_dataset(block_size = block_size)
    if hasattr(data_wrapper,"dataset_info"):
        dataset_train = data_wrapper.dataset_info.train
        dataset_eval = data_wrapper.dataset_info.eval
    else:
        dataset_train = data_wrapper.datasets.train
        dataset_eval = data_wrapper.datasets.eval

    try:
        if cfg.dataset.sizes.train >= 1.0:
            log.warning(f'Training on the entirety of the dataset without leaving some data for testing.')
    except:
        log.warning(f'Old Dataset definitions have not been updated completely')
        
    #check if batch_size batches are going to be performed
    from torch.utils.data import Dataset
    def check_dataset_size(name: str, dataset: Dataset):
        batch_size = cfg.dataset.dataloader.batch_size
        #model which is not an llm is loaded
        if cfg.dataset.args.get('block_size') is None:
            log.info(f'Model for dataset {name} presumably is not an LLM as the block size has not been specified')
            return
        block_size = cfg.dataset.args.block_size
        if batch_size * block_size > len(dataset):
            log.warning(f'The product of batch_size {batch_size} and block_size {block_size} is larger than the dataset {name}, causing the dataloader to skip batches. Maybe check the split size?')
    
    train_dataloader = None
    eval_dataloader = None
    bench_dataloader = None
    if isinstance(data_wrapper, OldDatasetWrapper):
        check_dataset_size("train", dataset_train)
        train_dataloader = get_loader(dataloader_cfg = cfg.dataset.dataloader, data = dataset_train)
        if dataset_eval is not None:
            check_dataset_size("eval", dataset_eval)
            eval_dataloader = get_loader(dataloader_cfg = cfg.dataset.dataloader, data = dataset_eval)
        else:
            eval_dataloader = None

        #update tokenizer config with metadata to save it in model checkpoints
        data_wrapper.tokenizer.load_metadata(filepath=os.path.join(data_wrapper.tokenized_dir, cfg.dataset.tokenizer.meta_file))
        with open_dict(cfg.dataset.tokenizer):
            cfg.dataset.tokenizer["meta"] = data_wrapper.tokenizer.meta
        
        max_token_value = data_wrapper.tokenizer.meta.max_token_value
        if max_token_value < cfg.model.args.vocab_size:
            log.warning(f'Vocab size of model is larger than the tokenizer vocab. vocab_size of model: {cfg.model.args.vocab_size}, vocab size of tokenizer {max_token_value}')

            #log.warning(f'Vocab size of model is larger than the tokenizer vocab. Setting vocab_size to: {max_token_value} to prevent errors during inference')
            #OmegaConf.update(cfg, "model.args.vocab_size", max_token_value, force_add=True)
    else:
        train_dataloader = data_wrapper.get_loader('train')
        eval_dataloader  = data_wrapper.get_loader('eval')
        bench_dataloader = data_wrapper.get_loader('bench')

        max_token_value = len(data_wrapper.tokenizer.get_vocab())
        log.info(f"number token ids in tokenizer {max_token_value}")
        # TODO find out what meta data does and ijmportance of max_token_value

    if bench_dataloader is not None:
        return (train_dataloader, eval_dataloader, bench_dataloader)
    else:
        return (train_dataloader, eval_dataloader)