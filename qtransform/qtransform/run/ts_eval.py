import json
import logging
from os.path import join, expanduser, exists
from os import makedirs
from huggingface_hub import hf_hub_download
from omegaconf import DictConfig

from qtransform.model import QTRModelWrapper
import os
import torch
from datetime import datetime
import yaml
from qtransform.run.infer import generate
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from qtransform import device_singleton
from qtransform.model import get_model_wrapper, DynamicCheckpointQTRModelWrapper
from functools import lru_cache
from tqdm import tqdm

log = logging.getLogger(__name__)


# Keep track of 10 different messages and then warn again
@lru_cache(1)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)


def run(cfg: DictConfig):
    """ launches energy benchmark with provided config"""
    log.info("================")
    log.info("Running TinyStories eval prompts generation")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")

    torch.manual_seed(cfg.seed)

    model_wrapper: DynamicCheckpointQTRModelWrapper = get_model_wrapper(cfg.model)
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
        # quantize last layers (batchnorm). params last saved checkpoint do not entirely reflect current model anymore
        # if replace_layers_later is not None:
        #    model, _ = quantizer.get_quantized_model(replace_layers_later)
    assert isinstance(model_wrapper,
                      DynamicCheckpointQTRModelWrapper), f'Model should be torch module, not {type(model_wrapper)}'
    # only parameters (type torch.nn.parameter.Parameter) are moved to the device, not non-named Tensors
    # this is a problem if a layer uses a non-named Tensor during the forward pass
    model_wrapper.to(device=device_singleton.device)
    if hasattr(log, "trace"): log.trace(model_wrapper.model)

    log.info(f"Starting completion of TinyStories eval prompts")
    # for now. This just prevent the error msg, maybe in the future we find a way of using the hf-tok-parallelism feature
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer_singleton.tokenizer = cfg.tokenizer

    file_path = hf_hub_download(repo_id="roneneldan/TinyStories",
                                repo_type="dataset",
                                filename="Evaluation prompts.yaml")

    with open(file_path, "r", encoding="utf-8") as file:
        prompts = yaml.safe_load(file)

    data = generation(cfg, model_wrapper, prompts)

    save_results(cfg, data)


def generation(cfg, model_wrapper: QTRModelWrapper, prompts: list[str]) -> list[dict[str, str]]:
    """
    max_new_tokens, max_iters, temperature and top_k can be set through the configs run parameters.
    """
    ins_outs = []
    with torch.no_grad():
        for prompt in tqdm(prompts):
            inputs = tokenizer_singleton.tokenizer.encode(prompt)
            inputs = torch.tensor(inputs, device=device_singleton.device).unsqueeze(dim=0)

            outputs: torch.Tensor = generate(model_wrapper=model_wrapper, idx=inputs,
                                             max_new_tokens=cfg.run.max_new_tokens,
                                             temperature=cfg.run.temperature,
                                             top_k=cfg.run.top_k)

            # print(tokenizer_singleton.tokenizer.decode(outputs[0].tolist()) + '\n---------------\n')
            ins_outs.append({
                "prompt": tokenizer_singleton.tokenizer.decode(inputs[0].tolist()),
                "completion": tokenizer_singleton.tokenizer.decode(outputs[0].tolist()[len(inputs[0]):])
            })

    return ins_outs


def save_results(cfg, data: list[dict[str, str]]):
    """
    Saves results to path specified in configs run parameters.
    """
    base_out_path = cfg.run.out.path
    if base_out_path:
        base_out_path = base_out_path.replace('~', expanduser('~'))
        if not exists(base_out_path):
            log.debug(f'Creating base output dir: {base_out_path}')
            makedirs(base_out_path, exist_ok=True)

    with open(join(base_out_path, "results.json"), "w") as outfile:
        json.dump(data, outfile, indent=4)
