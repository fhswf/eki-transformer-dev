import logging
from typing import Any, Tuple, Union
from omegaconf import DictConfig, OmegaConf, open_dict
from hydra import compose
from zeus.monitor import ZeusMonitor

from qtransform.model import gpt, ModelArgs, QTRModelWrapper
from transformers import AutoTokenizer
import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data as torch_data  # prevent naming conflict with data from dataloaders
from datetime import datetime
import torch.nn.functional as F
from time import time, sleep
from qtransform.utils.checkpoint import save_checkpoint
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from qtransform import device_singleton
from qtransform.model import get_model_wrapper, DynamicCheckpointQTRModelWrapper
from torch.profiler import profile, record_function, ProfilerActivity
from functools import lru_cache
from qtransform.wandb import wandb_watch, wandb_log

log = logging.getLogger(__name__)


# Keep track of 10 different messages and then warn again
@lru_cache(1)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)


def run(cfg: DictConfig):
    """ launches training with provided config"""
    log.info("================")
    log.info("Running energy benchmark")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")

    if "dataloader" not in cfg.dataset:
        log.error(
            f"dataloder not specified for dataset: {cfg.dataset.name}. Use dataset=huggingface to get one automaticly.")
    device_singleton.device = cfg.device
    device = device_singleton.device
    torch.manual_seed(cfg.seed)

    log.info(f"number of torch dataloader: {str(cfg.dataset.dataloader.num_workers)}")
    model_wrapper: DynamicCheckpointQTRModelWrapper = get_model_wrapper(cfg.model)
    # TODO: move quant_config as subconfig into model_cfg to perform quantization within modelwrapper
    quant_cfg = cfg.get('quantization')
    if quant_cfg and quant_cfg.quantize:
        if not model_wrapper.quantized:
            log.info(f'Quantizing model')
            model_wrapper.quantize_model(quant_cfg)
        else:
            warn_once(log, f'Model was already quantized, ignoring quant_cfg from hydra')
        # from qtransform.quantization import get_quantizer
        # quantizer, model_quant_cfg = get_quantizer(quant_cfg, model=model)
        # model, replace_layers_later = quantizer.get_quantized_model(model_quant_cfg, inplace=True)
        # quantize last layers (batchnorm). parmams last saved checkpoint do not entirely reflect current model anymore
        # if replace_layers_later is not None:
        #    model, _ = quantizer.get_quantized_model(replace_layers_later)
    assert isinstance(model_wrapper,
                      DynamicCheckpointQTRModelWrapper), f'Model should be torch module, not {type(model_wrapper)}'
    # only parameters (type torch.nn.parameter.Parameter) are moved to the device, not non-named Tensors
    # this is a problem if a layer uses a non-named Tensor during the forward pass
    model_wrapper.to(device=device)
    if hasattr(log, "trace"): log.trace(model_wrapper.model)


    log.info(f"Starting benchmark")
    # for now. This just prevent the error msg, maybe in the future we find a way of using the hf-tok-parallelism feature
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_singleton.tokenizer = cfg.tokenizer
    from qtransform.dataset import DataLoaderWrapper, DatasetSplitType
    dataloader_wrapper = DataLoaderWrapper(cfg.dataset)
    train_dataloader = dataloader_wrapper.get_loader(DatasetSplitType.TRAIN)
    eval_dataloader = dataloader_wrapper.get_loader(DatasetSplitType.EVAL)

    # cfg.run => energy.yaml
    monitor = ZeusMonitor(log_file="~/energy_measurement.csv")
    #device_str = str(device)
    #domains = None
    #if device_str == "cpu":
    #    domains = [RaplPackageDomain(0), RaplUncoreDomain(0)]
    #elif device_str == "cuda":
    #    domains = [NvidiaGPUDomain(0)]
    #else:
    #    raise ValueError(f"Unsupported device: {device}")

    bench(cfg, model_wrapper, eval_dataloader, tokenizer_singleton, monitor)

def bench(cfg, model_wrapper: QTRModelWrapper, dataloader, tokenizer_singleton, monitor : ZeusMonitor):

    lens = min(len(dataloader), cfg.run.max_iters)
    tokenizer = tokenizer_singleton.tokenizer
    if isinstance(model_wrapper.model, torch.nn.Module):
        model_wrapper.model.eval()
    log.info("Measuring energy consumption during generation")
    for i, data in enumerate(dataloader):

        log.debug(f'Iteration: {i}')
        if i >= lens:
            break
        inputs = None
        labels = None
        if len(data) > 2:
            inputs = data['input_ids']
            labels = data['labels']
        elif len(data) == 2:
            inputs, labels = data
        else:
            log.error(f"unsupported dataloader output. len was {len(data)}. ")
            raise NotImplementedError
        with torch.no_grad():
            inputs = inputs.to(device_singleton.device)
            labels = labels.to(device_singleton.device)

            monitor.begin_window("Inference")
            output = model_wrapper(inputs, labels)
            measurement = monitor.end_window("Inference", sync_execution=True)
            print(measurement.total_energy)
    log.info("Measurements finished")
