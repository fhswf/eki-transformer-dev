import logging
log = logging. getLogger(__name__)
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, open_dict
from . import forward_pass, get_dataloader_and_tokenizer, load_model, ModelData, InferType, generate
#TODO: make ... import compatible
from qtransform.model import get_model_wrapper, QTRModelWrapper
from qtransform import device_singleton
from qtransform.dataset import get_loader
from typing import List, Union, Tuple
from time import time
from datetime import datetime
from torch.profiler import profile, record_function, ProfilerActivity
from tabulate import tabulate

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

    # TODO
    #model = onnx.load("model.onnx")
    #input_shapes = [[d.dim_value for d in _input.type.tensor_type.shape.dim] for _input in model.graph.input]
    #TODO: infer model config (block size)
    row_limit = cfg.run.row_limit
    if not isinstance(row_limit, int):
        log.warning(f'row_limit should be a number. defaulting to 10.')
        row_limit = 10
    elif row_limit < 1:
        row_limit = 10
    from qtransform.model import QTRModelWrapper
    log.debug(f"Model config {cfg.model}")
    model_wrapper: QTRModelWrapper = get_model_wrapper(cfg.model)

    data_loader_tuples  = get_dataloader_and_tokenizer(cfg, model_wrapper.model_cfg.args.block_size)
    if len(data_loader_tuples) == 3:
        _, _, bench_dataloader  = data_loader_tuples
    elif len(data_loader_tuples) == 2:
        log.warning("using eval dataloader as benchmarking, as no bench_dataloader was provided")
        _, bench_dataloader = data_loader_tuples
    elif len(data_loader_tuples) == 1:
        log.warning("get_dataloader_and_tokenizer did only return one dataloader, be careful this dataset split was not used for training ")
        bench_dataloader = data_loader_tuples
    else:
        raise ValueError(f"To many dataloader where returned from 'get_dataloader_and_tokenizer'. Maybe redo this mapping?")

    if cfg.run.profile:
        #benchmark resource consumption (https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
        activities = ProfilerActivity.CPU if device.type == 'cpu' else ProfilerActivity.CUDA 
        with profile(activities=[activities], profile_memory=True, record_shapes=True) as prof:
            with record_function(f'BENCHMARK: {model_wrapper.model_type}'):
                log.info(f'Benchmark results: \n{benchmark(cfg=cfg, model_wrapper=model_wrapper, bench_dataloader=bench_dataloader)}')
        log.info(f'\n{prof.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit)}')
    else:
        log.info(f'BENCHMARK for : {model_wrapper.model.type.name}')
        log.info(f'Benchmark results: \n{benchmark(cfg=cfg, model=model_wrapper, bench_dataloader=bench_dataloader)}')

    """
    #load model
    #TODO: huggingface models are not able to be finetuned this way. implement saving of huggingface checkpoints (as well as their configs)
    if cfg.run.pretrained_model is not None:
        log.info(f'Using pretrained model {cfg.run.pretrained_model}')
        from qtransform.model.hf_gpt2 import PreTrainedGPT2
        from qtransform.dataset.tokenizer.tiktoken import TikTokenizer
        model = PreTrainedGPT2(DictConfig({"version": cfg.run.pretrained_model, "shift_targets": True})).to(device=device)
        #tokenizer = TikTokenizer({"encoding": "gpt2"})
        models: List[ModelData] = [ModelData(type = InferType.CHECKPOINT, 
                                        model = model, 
                                        #tokenizer = tokenizer, 
                                        name="hf-pretrained-"+cfg.run.pretrained_model,
                                        block_size=model.config.n_positions)]
    else:
        models: List[ModelData] = load_model(cfg, device)

        _, _, bench_dataloader = get_dataloader_and_tokenizer(cfg, models[0].model.config.block_size)

    #from qtransform.dataset import get_data, get_loader, DatasetWrapper
    #data_wrapper: DatasetWrapper = get_data(cfg.dataset)
    ##dataset hydra config expects block size, currently set in command line. TODO: infer from onnx metadata or checkpoint metadata
    #data_wrapper.load_dataset()
    #dataset_bench = data_wrapper.dataset_info.bench
    #if cfg.dataset.sizes.train >= 1.0:
    #    log.warning(f'Training on the entirety of the dataset without leaving some data for testing.')
    ##check if batch_size batches are going to be performed
    #from torch.utils.data import Dataset
    #def check_dataset_size(name: str, dataset: Dataset):
    #    batch_size = cfg.dataset.dataloader.batch_size
    #    #model which is not an llm is loaded
    #    if cfg.dataset.args.get('block_size') is None:
    #        log.info(f'Model for dataset {name} presumably is not an LLM as the block size has not been specified')
    #        return
    #    block_size = cfg.dataset.args.block_size
    #    if batch_size * block_size > len(dataset):
    #        log.warning(f'The product of batch_size {batch_size} and block_size {block_size} is larger than the dataset {name}, causing the dataloader to skip batches. Maybe check the split size?')
    #check_dataset_size("bench", dataset_bench)
    #bench_dataloader = get_loader(dataloader_cfg = cfg.dataset.dataloader, data = dataset_bench)
     
    if cfg.run.profile:
        #benchmark resource consumption (https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
        activities = ProfilerActivity.CPU if device.type == 'cpu' else ProfilerActivity.CUDA 
        for i, model_data in enumerate(models):
            with profile(activities=[activities], profile_memory=True, record_shapes=True) as prof:
                with record_function(f'BENCHMARK: {model_data.type.name}'):
                    log.info(f'Benchmark results: \n{benchmark(cfg=cfg, model_data=model_data, bench_dataloader=bench_dataloader)}')
            log.info(f'\n{prof.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit)}')
    else:
        for i, model_data in enumerate(models):
            log.info(f'BENCHMARK for : {model_data.type.name}')
            log.info(f'Benchmark results: \n{benchmark(cfg=cfg, model_data=model_data, bench_dataloader=bench_dataloader)}')"""
  
def benchmark(cfg, model_wrapper: QTRModelWrapper, bench_dataloader) -> Union[str, None]:
    print(model_wrapper.model)
    if "num_samples" not in cfg.run:
        lens = min(len(bench_dataloader), cfg.run.max_iters)
    else:
        lens = min(len(bench_dataloader), cfg.run.num_samples)

    log.info(f"Datalaoder has {len(bench_dataloader)} number of samples ")
    log.info(f"Running Benchmark for {lens} samples")
    log.warning(f"Datalaoder length might not be correct")

    # for logging only:
    running_loss = 0
    last_loss = 0
    batch_time = time()

    perplexity = torch.zeros(lens)
    accuracy = torch.zeros(lens)
    if isinstance(model_wrapper.model, torch.nn.Module):
        model_wrapper.model.eval()
    for i, data in enumerate(bench_dataloader): 
        if i >= lens:
            break
        inputs = None
        labels = None
        if len(data) > 2:
            inputs = data['input_ids']
            labels = data['labels']
            attention_mask = data['attention_mask']
        elif len(data) == 2:
            inputs, labels = data
        else:
            log.error(f"unsupported dataloader output. len was {len(data)}. ")
            raise NotImplementedError
        with torch.no_grad():
            inputs = inputs.to(device_singleton.device)
            labels = labels.to(device_singleton.device)

            cond_model_shifts_targets_internally = model_wrapper.model_cfg.args.get("shift_targets") and model_wrapper.model_cfg.get("calc_loss_in_model")
            cond_labels_eq_input = torch.all(inputs == labels).item()   # data loader supplies labels and inputs without shifted labels to the right => needs to be done by model

            # TODO maybe we can support this but then would would need 8 if else statements here
            if not model_wrapper.model_cfg.args.get("shift_targets") and model_wrapper.model_cfg.get("calc_loss_in_model"):
                log.error(f"unsupported combination of model args shift_targets {model_wrapper.model_cfg.args.get('shift_targets')} and calc_loss_in_model {model_wrapper.model_cfg.get('calc_loss_in_model')}")
                raise ValueError(f"unsupported combination of model args shift_targets {model_wrapper.model_cfg.args.get('shift_targets')} and calc_loss_in_model {model_wrapper.model_cfg.get('calc_loss_in_model')}")
            if model_wrapper.model_cfg.args.get("shift_targets") and not model_wrapper.model_cfg.get("calc_loss_in_model"):
                log.error(f"unsupported combination of model args shift_targets {model_wrapper.model_cfg.args.get('shift_targets')} and calc_loss_in_model {model_wrapper.model_cfg.get('calc_loss_in_model')}")
                raise ValueError(f"unsupported combination of model args shift_targets {model_wrapper.model_cfg.args.get('shift_targets')} and calc_loss_in_model {model_wrapper.model_cfg.get('calc_loss_in_model')}")
          
            logits = None
            loss = None
            if hasattr(log,"trace"): log.trace(f"cond_model_shifts_targets_internally {cond_model_shifts_targets_internally}")
            if hasattr(log,"trace"): log.trace(f"cond_labels_eq_input {cond_labels_eq_input}")
            #print(cond_model_shifts_targets_internally)
            #print(cond_labels_eq_input)
            if cond_model_shifts_targets_internally and cond_labels_eq_input:
                output = model_wrapper(inputs, labels)
                if not isinstance(output, tuple):
                    raise ValueError(f"as model shifts internally and computes loss, model output needs to be a tuple")
                logits, loss = output
                
            elif not cond_model_shifts_targets_internally and cond_labels_eq_input: 
                # here we shit labels our selfs and only feed inputs to the model, potentially just ignore labels and loss if returned
                output = model_wrapper(inputs, None)
                if isinstance(output, tuple):
                    logits = output[0]
                else:
                    logits = output
                # print("=======")
                # torch.set_printoptions(edgeitems=5000)
                # print(logits[0][0])
                # torch.set_printoptions(edgeitems=3)
                # print(logits.size())
                # print(labels.size())
                # print(logits)
                # print(labels)
                # thats the shift
                logits = logits[..., :-1, :].contiguous()
                labels = labels[..., 1:].contiguous()
                # print("---------")
                # print(logits)
                # print(labels)
                # get regular crossentropy loss
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
            elif not cond_model_shifts_targets_internally and not cond_labels_eq_input:
                pass
            elif cond_model_shifts_targets_internally and not cond_labels_eq_input:
                raise NotImplementedError(f"cond_model_shifts_targets_internally {cond_model_shifts_targets_internally} and cond_labels_eq_input {cond_labels_eq_input}")
            else:
                raise NotImplementedError(f"cond_model_shifts_targets_internally {cond_model_shifts_targets_internally} and cond_labels_eq_input {cond_labels_eq_input}")

            probs = F.softmax(logits, dim=-1)
            accuracy[i] = measure_accuracy(model_wrapper.model, labels=labels, inputs=probs)
            perplexity[i] = torch.exp(loss)
            #print(f'Loss: {loss}, Perplexity: {torch.exp(loss)}, Acc: {accuracy[i]}')
            log.debug(f'Loss: {loss}, Perplexity: {torch.exp(loss)}, Acc: {accuracy[i]}')
            
            #log loss
            running_loss += loss.item()
            if i % cfg.run.log_steps_interval == cfg.run.log_steps_interval-1:
                last_loss = running_loss / cfg.run.log_steps_interval # loss per batch
                log.info(f'  batch {i} Loss: {last_loss}, Perplexity: {torch.exp(loss)}, Acc: {accuracy[i]}. time: {(time() - batch_time)*1000:.2f}ms')
                batch_time = time()
                running_loss = 0

        #other_perplexity += measure_perplexity(probs, labels)
        #other_accuracy += measure_accuracy(model_type=model_data.type, model=model_data.model, labels=labels, inputs=probs)
        torch.cuda.empty_cache()
    #table printing from: https://learnpython.com/blog/print-table-in-python/
    #Benchmarking columns are derived from the attention is all you need paper (https://arxiv.org/pdf/1706.03762.pdf, page 9)
    return tabulate([['path', 'avg_ppl', 'acc_in_%'],[model_wrapper.model_type, perplexity.mean(), accuracy.mean()]],headers='firstrow', tablefmt='simple_grid')

"""
calculating the perplexity usually occurs with an input of smaller size than the model's max context length
(https://huggingface.co/docs/transformers/perplexity#calculating-ppl-with-fixed-length-models).
unsure if the perplexity of a model is different when the same sample is forwarded in form of smaller chunks as opposed
to the entire sample at once
"""
def measure_perplexity(logits: torch.Tensor, labels: torch.Tensor):
    #cross entropy either expects the probabilities of tokens or a list of tokens
    #(https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html)
    result = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-1)
    return  torch.exp(result)#F.cross_entropy(logits, labels))

def measure_accuracy(model_wrapper: QTRModelWrapper, labels: torch.Tensor, inputs: torch.Tensor = None) -> float:
    """
    Measures how many token predictions of a logit and label are correct. It does so by either generating text based on the start of the label tensor and then
    comparing the result with the actual values within labels or by directly comparing the results if inputs is specified.
    
    Arguments: inputs: softmaxed probability distribution of words 
               labels: actually occuring words

    Inputs is either of shape: [N, C, V] or [C, V] with N: batches, C: context, V: vocab_size
    Labels is either of shape: [N, C] or [C] with N: batches, C: context
    """
    #unsure if measuring accuracy without labels is needed
    """if inputs is None:
        log.warning(f'Not changed for QTRModelWrapper yet')
        #input contains complete batches of samples from dataset, cut a small portion (at max half of tokens) and generate text
        N, C = labels.size()
        device = labels.device
        prompt_length = torch.randint(low=1, high=C//2, size=(1,)).item()
        log.debug(f'Begin measuring accuracy with prompt_length: {prompt_length}')
        accuracy_batch = torch.zeros(N).to(device=device)
        for i, batch in enumerate(labels):
            #generate function expects a batch size of 1
            #(model_type: InferType, block_size: int, model: Union[nn.Module, ModelWrapper], idx: torch.Tensor, max_new_tokens: int, temperature: float =1.0, top_k: int =None):
            logits = generate(model_type, block_size = 256, model = model, idx = batch[:prompt_length].unsqueeze(dim=0), max_new_tokens = C - prompt_length)
            #log.critical(f'{logits}, {labels[i].unsqueeze(dim=0)}')
            #then, compare tokens with label
            _, accuracy = labels[i].unsqueeze(dim=0).eq(logits).unique(return_counts=True)
            log.debug(f'{accuracy}')
            #input prompt is always going to be correct
            accuracy = (accuracy[1] - prompt_length) / (C - prompt_length) * 100
            accuracy_batch[i] = accuracy
        return accuracy_batch.mean().item()"""
    #entries ordered as: False, True
    _, accuracy =  inputs.max(dim=-1).indices.eq(labels).unique(return_counts=True)
    #if length is one, then accuracy only contains false values
    accuracy = accuracy[1] / (accuracy[0] + accuracy[1]) * 100 if len(accuracy) == 2 else 0.0
    return accuracy
