from typing import Tuple, Union, List, Dict, Any
from omegaconf import DictConfig, open_dict
import os
from dataclasses import dataclass
from qtransform.dataset.tokenizer.tokenizer import Tokenizer
from qtransform.dataset import tokenizer as tokenizer_module
from qtransform.utils.helper import load_checkpoint, load_onnx_model
from qtransform.utils.introspection import get_classes
from qtransform.model import QTRModelWrapper, ModelType
import torch
from torch import nn
import torch.nn.functional as F
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
from enum import Enum
from logging import getLogger

log = getLogger(__name__)

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
    for _ in range(max_new_tokens):
        # if the sequence context is growing too long we must crop it at block_size
        block_size = model_wrapper.model_cfg.args.block_size
        idx_cond = idx if idx.size(1) <= block_size else idx[:, -block_size:]
        # forward the model to get the logits for the index in the sequence
        # the results should not be softmaxed yet as they will be later within this function
        logits, _ = model_wrapper(idx_cond)
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
    """ note that the tokenizer is inside the dataloader wrapper  ...  for now """

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