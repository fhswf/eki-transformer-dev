import logging
log = logging. getLogger(__name__)
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from . import forward_pass, load_model, ModelData, InferType, generate
from qtransform import device_singleton
from qtransform.dataset import get_loader
from typing import Union, Tuple
from datetime import datetime

TABLE_HEADERS = ['Model', 'n_transformer_blocks', 'n_attn_heads', 'embd_dim', 'Train steps', 'Accuracy', 'PPL', 'params']

def run(cfg : DictConfig):
    #everything below is a copy paste from the training script. TODO: generalise run scripts
    log.info("================")
    log.info("Running Benchmarking")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")

    if "dataloader" not in cfg.dataset:
        log.error(f"dataloder not specified for dataset: {cfg.dataset.name}. Use dataset=huggingface to get one automaticly.")

    device_singleton.device = cfg.device
    device = device_singleton.device
    if device.type == 'cuda':
        cuda_kwargs = {'pin_memory': True,}
        #struct flag of dictconf prevents additional keys to be added (https://omegaconf.readthedocs.io/en/latest/usage.html#struct-flag)
        with open_dict(cfg.dataset.dataloader):
            cfg.dataset.dataloader.update(cuda_kwargs)
    torch.manual_seed(cfg.seed)    
    log.info(f"number of torch dataloader: {str(cfg.dataset.dataloader.num_workers)}")

    from qtransform.dataset import get_data, get_loader, DatasetWrapper
    data_wrapper: DatasetWrapper = get_data(cfg.dataset)
    #dataset hydra config expects block size, currently set in command line. TODO: infer from onnx metadata or checkpoint metadata
    data_wrapper.load_dataset()
    dataset_bench = data_wrapper.dataset_info.bench
    if cfg.dataset.sizes.train >= 1.0:
        log.warning(f'Training on the entirety of the dataset without leaving some data for testing.')
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
    check_dataset_size("bench", dataset_bench)
    bench_dataloader = get_loader(dataloader_cfg = cfg.dataset.dataloader, data = dataset_bench)
    # copy paste ends here

    #TODO: infer model config (block size)

    #load model
    models: List[ModelData] = load_model(cfg, device)
    accuracy_models = torch.zeros(len(models), cfg.run.num_samples)
    perplexity = torch.zeros(len(models), cfg.run.num_samples)
    accuracy = torch.zeros(len(models), cfg.run.num_samples)
    for i, model_data in enumerate(models):
        if isinstance(model_data.model, torch.nn.Module):
            model_data.model.eval()
            #TODO: add training steps in checkpoint data, create meta file for onnx models containing tokenizer data and model structure
            #meta = model_data.model.config
        if model_data.type != InferType.CHECKPOINT:
            log.warning(f'Benchmarking for ONNX models not implemented yet.')
            continue
        for j, data in enumerate(bench_dataloader):
            if j >= cfg.run.num_samples:
                break
            inputs, labels = data
            inputs = inputs.to(device_singleton.device)
            labels = labels.to(device_singleton.device)
            output = forward_pass(model_data.type, model_data.model, inputs)
            if isinstance(output, tuple):
                logits = outputs[0]
            else:
                logits = output
            probs = F.softmax(logits, dim=-1)
            perplexity[i][j] = measure_perplexity(probs, labels)
            accuracy[i][j] = measure_accuracy(model_type=model_data.type, model=model_data.model, labels=labels, inputs=probs, device=device)
        #table printing from: https://learnpython.com/blog/print-table-in-python/
        #Benchmarking columns are derived from the attention is all you need paper (https://arxiv.org/pdf/1706.03762.pdf, page 9)
        from tabulate import tabulate
        print(tabulate([['path', 'avg_ppl', 'acc'],[model_data.name, perplexity.mean(), accuracy.mean()]],headers='firstrow', tablefmt='simple_grid'))


    
"""
calculating the perplexity usually occurs with an input of smaller size than the model's max context length
(https://huggingface.co/docs/transformers/perplexity#calculating-ppl-with-fixed-length-models).
unsure if the perplexity of a model is different when the same sample is forwarded in form of smaller chunks as opposed
to the entire sample at once
"""
def measure_perplexity(logits: torch.Tensor, labels: torch.Tensor):
    #cross entropy either expects the probabilities of tokens or a list of tokens
    #(https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
    return torch.exp(F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1))#F.cross_entropy(logits, labels))

def measure_accuracy(model_type: InferType, model, labels: torch.Tensor, inputs: torch.Tensor = None, device: torch.device = None) -> float:
    """
    Measures how many token predictions of a logit and label are correct. It does so by either generating text based on the start of the label tensor and then
    comparing the result with the actual values within labels or by directly comparing the results if inputs is specified.
    
    Arguments: inputs: softmaxed probability distribution of words 
               labels: actually occuring words

    Inputs is either of shape: [N, C, V] or [C, V] with N: batches, C: context, V: vocab_size
    Labels is either of shape: [N, C] or [C] with N: batches, C: context
    """
    if inputs is None:
        #input contains complete batches of samples from dataset, cut a small portion (at max half of tokens) and generate text
        N, C = labels.size()
        prompt_length = torch.randint(low=1, high=C//2, size=(1,)).item()
        log.debug(f'Begin measuring accuracy with prompt_length: {prompt_length}')
        accuracy_batch = torch.zeros(N).to(device=device)
        for i, batch in enumerate(labels):
            #generate function expects a batch size of 1
            logits = generate(model_type, model, batch[:prompt_length].unsqueeze(dim=0), max_new_tokens = C - prompt_length)
            #log.critical(f'{logits}, {labels[i].unsqueeze(dim=0)}')
            #then, compare tokens with label
            _, accuracy = labels[i].unsqueeze(dim=0).eq(logits).unique(return_counts=True)
            log.debug(f'{accuracy}')
            #input prompt is always going to be correct
            accuracy = (accuracy[1] - prompt_length) / (C - prompt_length) * 100
            accuracy_batch[i] = accuracy
        return accuracy_batch.mean().item()
    #entries ordered as: False, True
    _, accuracy =  inputs.max(dim=-1).indices.eq(labels).unique(return_counts=True)
    #if length is one, then accuracy only contains false values
    accuracy = accuracy[1] / (accuracy[0] + accuracy[1]) * 100 if len(accuracy) == 2 else 0.0
    return accuracy

from qtransform.quantization.quant_bn import replace_bn, QuantBatchnorm1d, CustomBatchNorm1d
from qtransform.quantization.brevitas_quant import BrevitasQuantizer
from qtransform.quantization import LayerQuantConfig
"""
TODO: not sure at what step batchnorm layers should be merged.
      if batchnorm is merged before training while training from scratch, the default values would be merged 
      if batchnorm is merged before training during ptq, it would make sense but batchnorm could potentially stay in the model
      if batchnorm is merged before training during ptq, the qparams would be learned during training 
      if batchnorm is merged after training during qat, the trained values from batchnorm would be merged but the qparams would be default
      if batchnorm is merged after training during ptq, qparams would still have their default values

      based on this, it would make the most sense to merge before training during ptq
      that could make the previous quantization during qat possibly redundant

"""
def compare_bn():
    """
    Compares the custom implementation of BatchNorm1d within qtransform with torch's batchnorm.
    TODO: it would be best to pass a BatchNorm layer that had its gamma and beta tensors trained
          to have that, a transformer model would have to be trained
    """
    #inputs comparable to a small gpt2 model for fpgas during training
    n,c,l = (12,64, 256)
    size = torch.Size([n,c,l])
    torch_bn = torch.nn.BatchNorm1d(c)
    torch_bn.train()
    iters = 100
    #feed some dummy values to adjust the mean and standard deviation
    compare_loss(torch_bn, torch.nn.Identity(), size, iters)
    custom_bn = CustomBatchNorm1d(c)
    custom_bn = replace_bn(bn=torch_bn, new_bn=custom_bn, qat=False)
    #make some space for more results
    loss_fn = lambda x: compare_loss(torch_bn, custom_bn, size, 10).unsqueeze(0)
    result = compare_loss(torch_bn, custom_bn, size, 10).unsqueeze(0)
    torch_bn.eval()
    #TODO: actually benchmark this on a trained model
    for i in range(iters):
        result = torch.cat((result, compare_loss(torch_bn, custom_bn, size, 10).unsqueeze(0)), dim=0)
    log.info(f'Average loss when merging batchnorm: {result.mean()}')


def compare_loss(layer1: torch.nn.Module, layer2: torch.nn.Module, shape: torch.Size, iters: int) -> torch.Tensor:
    """
    Compares the result of two layers by passing random values with a specified shape into both layers and 
    calculating the mean from the difference of their outputs (out_layer1 - out_layer2). 
    This process is repeated iters times, returning a 1d Tensor which contains the results.

    This function should be useful for comparing the accuracy of outputs for quantized and non-quantized layers or
    custom implementations of layers such as MultiheadAttention, BatchNorm, LayerNorm etc.
    """
    result: torch.Tensor = torch.zeros(iters)
    for i in range(iters):
        #random input: TODO: maybe load dataset or something
        input = torch.randn(shape)
        out_l1 = layer1(input)
        out_l2 = layer2(input)
        result[i] = (out_l1 - out_l2).abs().mean()
    return result