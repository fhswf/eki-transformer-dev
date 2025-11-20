import logging
from typing import Any, Tuple, Union
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data as torch_data #prevent naming conflict with data from dataloaders
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime, timedelta
import torch.nn.functional as F
from pprint import PrettyPrinter
from time import time
import subprocess
import re
from qtransform.utils import save_checkpoint
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from qtransform import device_singleton
from qtransform.utils.helper import load_state_dict_proxy
from qtransform.model import QTRModelWrapper, get_model_wrapper, DynamicCheckpointQTRModelWrapper
from qtransform import ConfigSingleton
from torch.profiler import profile, record_function, ProfilerActivity
from functools import lru_cache
from qtransform.wandb import wandb_watch, wandb_log

log = logging.getLogger(__name__)

# Keep track of 10 different messages and then warn again
@lru_cache(1)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)

def get_slurm_job_info():
    """Get SLURM job information including time limits"""
    slurm_job_id = os.environ.get('SLURM_JOB_ID')
    if not slurm_job_id:
        return None
    
    try:
        # Get job information using scontrol
        result = subprocess.run(['scontrol', 'show', 'job', slurm_job_id], 
                              capture_output=True, text=True, check=True)
        output = result.stdout
        
        # Parse time limit (format: days-hours:minutes:seconds or hours:minutes:seconds)
        time_limit_match = re.search(r'TimeLimit=(\S+)', output)
        if not time_limit_match:
            return None
            
        time_limit_str = time_limit_match.group(1)
        
        # Parse start time
        start_time_match = re.search(r'StartTime=(\S+)', output)
        if not start_time_match:
            return None
            
        start_time_str = start_time_match.group(1)
        
        # Convert time limit to seconds
        if time_limit_str == 'UNLIMITED':
            time_limit_seconds = float('inf')
        else:
            time_limit_seconds = parse_slurm_time(time_limit_str)
            
        # Parse start time to datetime
        start_time = datetime.strptime(start_time_str, '%Y-%m-%dT%H:%M:%S')
        
        return {
            'job_id': slurm_job_id,
            'time_limit_seconds': time_limit_seconds,
            'start_time': start_time,
            'time_limit_str': time_limit_str
        }
    except (subprocess.CalledProcessError, ValueError, AttributeError) as e:
        log.warning(f"Failed to get SLURM job info: {e}")
        return None

def parse_slurm_time(time_str):
    """Parse SLURM time format to seconds"""
    if '-' in time_str:
        # Format: days-hours:minutes:seconds
        days_part, time_part = time_str.split('-', 1)
        days = int(days_part)
        hours, minutes, seconds = map(int, time_part.split(':'))
        return days * 86400 + hours * 3600 + minutes * 60 + seconds
    else:
        # Format: hours:minutes:seconds
        parts = time_str.split(':')
        if len(parts) == 3:
            hours, minutes, seconds = map(int, parts)
            return hours * 3600 + minutes * 60 + seconds
        elif len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
        else:
            return int(parts[0]) * 60  # assume minutes

def should_stop_training(slurm_info, buffer_minutes=10):
    """Check if training should stop due to approaching SLURM time limit"""
    if not slurm_info or slurm_info['time_limit_seconds'] == float('inf'):
        return False
        
    current_time = datetime.now()
    elapsed_seconds = (current_time - slurm_info['start_time']).total_seconds()
    remaining_seconds = slurm_info['time_limit_seconds'] - elapsed_seconds
    buffer_seconds = buffer_minutes * 60
    
    return remaining_seconds <= buffer_seconds

def get_remaining_time_info(slurm_info):
    """Get remaining time information for logging"""
    if not slurm_info or slurm_info['time_limit_seconds'] == float('inf'):
        return None
        
    current_time = datetime.now()
    elapsed_seconds = (current_time - slurm_info['start_time']).total_seconds()
    remaining_seconds = slurm_info['time_limit_seconds'] - elapsed_seconds
    
    elapsed_str = str(timedelta(seconds=int(elapsed_seconds)))
    remaining_str = str(timedelta(seconds=int(remaining_seconds)))
    
    return {
        'elapsed': elapsed_str,
        'remaining': remaining_str,
        'remaining_seconds': remaining_seconds
    }

def run(cfg: DictConfig):
    """ launches training with provided config"""
    log.info("================")
    log.info("Running Training")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")
    
    # Check if running in SLURM and get job information
    slurm_info = get_slurm_job_info()
    if slurm_info:
        log.info(f"Running in SLURM job {slurm_info['job_id']}")
        log.info(f"Job time limit: {slurm_info['time_limit_str']}")
        log.info(f"Job started at: {slurm_info['start_time']}")
        if slurm_info['time_limit_seconds'] != float('inf'):
            end_time = slurm_info['start_time'] + timedelta(seconds=slurm_info['time_limit_seconds'])
            log.info(f"Job will end at: {end_time}")
    else:
        log.info("Not running in SLURM environment")
    
    torch.autograd.set_detect_anomaly(True)
    #torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    ## note: float16 data type will automatically use a GradScaler
    #ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]

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
    model_wrapper: DynamicCheckpointQTRModelWrapper = get_model_wrapper(cfg.model)
    #TODO: move quant_config as subconfig into model_cfg to perform quantization within modelwrapper
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
    model_wrapper.to(device=device)
    if hasattr(log,"trace"): log.trace(model_wrapper.model)
    
    if model_wrapper.epochs >= 1:
        cfg.run.epochs = model_wrapper.epochs + cfg.run.epochs
        #TODO: construct absolute filepath for checkpoint
        log.info(f"Resuming training from: {cfg.model.from_file}")
        log.info(f"Epoch is {model_wrapper.epochs}, running for {cfg.run.epochs}")
    else:
        log.info(f"Starting new training")

    #elif "from_pretrained" in cfg.run and isinstance(cfg.run.from_pretrained, str):
    #    log.info(f"Loading model state dict from {cfg.run.from_pretrained}")
    #    from qtransform.model.gpt import GPT
    #    if not isinstance(model, GPT):
    #        log.error("from from_pretrained only works for GPT style model for now")
    #        raise Exception
    #    model = GPT.from_pretrained(model=model, model_type=cfg.run.from_pretrained)
    #    epochs_to_run = range(1, cfg.run.epochs + 1)

    # for now. This just prevent the error msg, maybe in the future we find a way of using the hf-tok-parallelism feature
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    tokenizer_singleton.tokenizer = cfg.tokenizer
    from qtransform.dataset import DataLoaderWrapper, DatasetSplitType
    dataloader_wrapper = DataLoaderWrapper(cfg.dataset)
    train_dataloader = dataloader_wrapper.get_loader(DatasetSplitType.TRAIN)
    eval_dataloader = dataloader_wrapper.get_loader(DatasetSplitType.EVAL)
    
    from qtransform.optim import get_optim, get_scheduler
    log.debug(f"optim config: {cfg.optim}")
    #optimizer = optim.Adadelta(model.parameters(), lr=cfg.optim.learning_rate)
    optimizer = get_optim(model=model_wrapper.model, optim_cfg=cfg.optim)
    log.debug(f'Configured optimizer ({type(optimizer)}): {optimizer}')
    scheduler = get_scheduler(optimizer=optimizer, scheduler_cfg = cfg.optim.scheduler)
    log.debug(f'Scheduler: {scheduler}')
    last_checkpoint = None
    # lets go
    last_checkpoint = train(
        cfg=cfg, 
        device=device, 
        model_wrapper=model_wrapper,
        train_data_loader=train_dataloader, 
        eval_data_loader=eval_dataloader,
        optimizer=optimizer, 
        scheduler=scheduler, 
        timestamp=timestamp,
        slurm_info=slurm_info
    )
    # maybe subsequent jobs can be managed by hydra in the future?
    # when this paradigm comes up more frequently we have to make this a thing ....
    log.debug("Finished training model")
    #update from_file for next run
    with open_dict(cfg):
        log.info(f'Updating from_file for rerun')
        cfg.model.from_file.filename = last_checkpoint
        cfg.model.from_file.model_dir = None
        
    if cfg.run.get("export") and not last_checkpoint:
        log.error(f"Cannot export model, no checkpoint saved during training {last_checkpoint=}")
        
    if cfg.run.get("export") and last_checkpoint:
        from qtransform.run import export
        from hydra import compose
        #load another entire hydra config with run=export, then override the current run config with export
        #this saves having to re-initialize the globalhydra configuration and further redundant config steps
        #(https://hydra.cc/docs/advanced/compose_api/ and https://github.com/facebookresearch/hydra/issues/440)
        export_cfg = compose(config_name="config", overrides=["run=export"])
        with open_dict(cfg):
            cfg.run = export_cfg.run
        OmegaConf.update(cfg, "model.from_file.filename", last_checkpoint+".onnx", force_add=True)
        OmegaConf.update(cfg, "model.from_file.model_dir", None, force_add=True)
        OmegaConf.update(cfg, "run.running_model", True, force_add=True)
        if model_wrapper.quantized:
            OmegaConf.update(cfg, "run.export_fn", "qonnx", force_add=True)
        else:
            OmegaConf.update(cfg, "run.export_fn", "onnx", force_add=True)
        kwargs = {"model_wrapper": model_wrapper}
        export.run(cfg, **kwargs)

def train(model_wrapper: DynamicCheckpointQTRModelWrapper, cfg: DictConfig, device, train_data_loader: torch_data.DataLoader, eval_data_loader: torch_data.DataLoader,
           optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler, timestamp: datetime, slurm_info: dict = None) -> Any:
    """ training over epochs with periodic logging and saving"""
    #print(model)
    mini_run = False
    epochs_to_run = None
    last_checkpoint = None
    if cfg.run.epochs == 0:
        cfg.run["epochs"] = 1
        warn_once(log, f"cfg.run.epochs is 0, performing mini training dry run")
        mini_run = True
    
    epochs_to_run = range(model_wrapper.epochs + 1, cfg.run.epochs + 1)
    model = model_wrapper.model
    
    # watch model parameters
    wandb_watch(models = model, log='all', log_freq=50, log_graph=True)
    wandb_log_metrics = {}
    
    if eval_data_loader is None:
        warn_once(log, f"Not running eval. Eval Dataloader is None")

    if cfg.optim.scheduler.warmup_epochs > epochs_to_run.stop -1:
        warn_once(log, f'Warmup epochs are larger than epochs to run, causing scheduler to never adjust learning rate.')
    # training loop
    for epoch in epochs_to_run:
            
        log.info(f"EPOCH: {epoch}/{cfg.run.epochs}")
        #dataloader always returns the same tensors after each epoch because it is casted inside of function call
        #therefore, cast it before training
        #TODO: find a more elegant solution, maybe by manipulating its seed with a torch.Generator?
        if cfg.run.profile.active:
            activities = [ProfilerActivity.CPU]
            if device.type == 'cuda':
                activities.append(ProfilerActivity.CUDA)
            row_limit = cfg.run.profile.row_limit
            if not isinstance(row_limit, int):
                row_limit = 10
            with profile(activities=activities, **cfg.run.profile.args) as prof:
                with record_function(f'TRAIN EPOCH {epoch}'):
                    last_loss, train_steps, early_stop = train_one_epoch(cfg, device, model, train_data_loader, optimizer, mini_run, eval_data_loader, epoch, timestamp, scheduler, slurm_info)
            log.info(f'\n{prof.key_averages().table(sort_by="cpu_time_total", row_limit=row_limit)}')
        else:
            last_loss, train_steps, early_stop = train_one_epoch(cfg, device, model, train_data_loader, optimizer, mini_run, eval_data_loader, epoch, timestamp, scheduler, slurm_info)
        
        # Check if early stopping was triggered
        if early_stop:
            log.warning("Early stopping triggered during epoch due to SLURM time limit")
            break

        if epoch % cfg.run.eval_epoch_interval == cfg.run.eval_epoch_interval-1 and eval_data_loader is not None:
            losses, mean = eval_model(cfg, device, model, eval_data_loader)
            log.info(f'AVERAGE EVAL LOSS FOR EPOCH {epoch}/{cfg.run.epochs}: {mean.item()}')
        
        # log for console
        log.info(f"last train loss was {str(last_loss)}")
        
        # logs for wandb
        wandb_log_metrics["validate/loss"] = mean.item()
        wandb_log_metrics["train/loss"] = last_loss
        wandb_log_metrics["train/batch"] = train_steps
        wandb_log_metrics["train/num_samples"] = train_steps * cfg.dataset.dataloader.batch_size
        wandb_log(wandb_log_metrics)
        
        if epoch % cfg.run.save_epoch_interval == 0 or epoch % cfg.run.epochs == 0: 
            ## interval or end of training, epochs is also 1 for mini_run
            # last_checkpoint is the absolute filepath of the saved checkpoint
            last_checkpoint: str = save_checkpoint(
                model=model, 
                optimizer=optimizer, 
                timestamp=timestamp, 
                epoch=epoch, 
                metrics=last_loss, 
                steps=train_steps * cfg.dataset.dataloader.batch_size)
            
        # Check if we should stop due to SLURM time limit
        buffer_minutes = cfg.run.get('slurm_time_buffer_minutes', 10)  # Default 10 minutes buffer
        if slurm_info and should_stop_training(slurm_info, buffer_minutes):
            log.warning("Approaching SLURM time limit. Stopping training early...")
            last_checkpoint: str = save_checkpoint(
                model=model, 
                optimizer=optimizer, 
                timestamp=timestamp, 
                epoch=epoch, 
                metrics=last_loss, 
                steps=train_steps * cfg.dataset.dataloader.batch_size)
            return last_checkpoint
        
        # advance learning rate
        if cfg.run.scheduler_step_type == 'epoch':
            if scheduler is not None:
                scheduler.step()
                new_lr = scheduler.get_last_lr()[0]
                log.info(f'New learning rate: {new_lr}')
    return last_checkpoint

def train_one_epoch(cfg: DictConfig, 
        device, 
        model: nn.Module, 
        train_data: Union[torch_data.DataLoader,torch_data.dataloader._MultiProcessingDataLoaderIter],
        optimizer: optim.Optimizer, 
        mini_run: bool=False, 
        eval_data_loader:Union[torch_data.DataLoader,torch_data.dataloader._MultiProcessingDataLoaderIter] = None,
        epoch: int = -1,
        timestamp = None,
        scheduler = None,
        slurm_info: dict = None,
        ) -> Tuple[Any, int, bool]:
    """ training loop over steps/batches """
    model.train() #if it was quantized, it could have been set to eval
    last_loss = 0
    running_loss = 0
    #cfg is entire hydra config
    gradient_accumulation_steps = cfg.run.get('gradient_accumulation_steps', 1)
    #dataloader already iterable, refer to TODO from train function for randomness in samples
    #if isinstance(train_data, torch_data.DataLoader):
    #    log.debug(f'Casting dataloader to iterable.')
    #    train_data: torch_data.dataloader._MultiProcessingDataLoaderIter = iter(train_data)
    if not isinstance(gradient_accumulation_steps, int):
        gradient_accumulation_steps = 1
    train_batch_time = time()
    #if max_iters is not specified, iterate through entire dataset
    if "max_iters" in  cfg.run and cfg.run.max_iters is not None and cfg.run.max_iters > 0:
        if cfg.run.max_iters < cfg.run.log_steps_interval:
            cfg.run.log_steps_interval = cfg.run.max_iters
        max_len = len(train_data)
        run_len = min(max_len, cfg.run.max_iters)
    else:
        max_len = len(train_data)
        run_len = len(train_data)
    log.info(f"train_data len is {max_len}, max_iters set to {cfg.run.get('max_iters', None)}. Running training for {run_len}")
    #for i in range(1, cfg.run.max_iters+1):
    for i, data in enumerate(train_data, 1):
        # Check SLURM time limit every 10 batches to avoid too frequent checks
        if slurm_info and i % 10 == 0:
            time_info = get_remaining_time_info(slurm_info)
            if time_info and i % 100 == 0:  # Log time info every 100 batches
                log.info(f"SLURM job time - Elapsed: {time_info['elapsed']}, Remaining: {time_info['remaining']}")
            
            buffer_minutes = cfg.run.get('slurm_time_buffer_minutes', 10)  # Default 10 minutes buffer
            if should_stop_training(slurm_info, buffer_minutes):
                log.warning(f"SLURM time limit approaching at batch {i}. Stopping training...")
                return last_loss, i, True  # Return early stop flag
            
        #log.critical(tokenizer_singleton.tokenizer.decode(data[0].tolist()))
        #print(data)
        if i > run_len:
            break
        loss = None #remember last loss of mini-batch
        #break one iteration down into multiple batches to simulate a larger batch size
        for micro_step in range(gradient_accumulation_steps):
            #dataloaders iterate through entire dataset
            #problematic if datasets are large (openwebtext ~21GB) and testing should be done
            #data = next(train_data)
            #token tensor of length block_size (context length)
            #print(data)
            inputs = None
            labels = None
            # dataloader from hf with additional attention_mask for 
            if len(data) > 2:
                inputs = data['input_ids']
                #log.info(tokenizer_singleton.tokenizer.decode(inputs[0].tolist())) #debug to make sure inputs make sense
                labels = data['labels']
                attention_mask = data['attention_mask']
                if not torch.all(attention_mask == 1):
                    log.error(f"mask for {i} is not all 1, BUT MASK IS IGNORED ATM")
                    log.error(data)
            elif len(data) == 2:
                inputs, labels = data
            else:
                log.error(f"unsupported dataloader output. len was {len(data)}. ")
                raise NotImplementedError
            inputs = inputs.to(device_singleton.device)
            labels = labels.to(device_singleton.device)            
            if cfg.model.calc_loss_in_model:
                if "shift_targets" in model.config.__dict__.keys() and not model.config.shift_targets:
                    warn_once(log, f"model does not shift_targets accoring to config but calculates loss inside model, this might not work")
                outputs, loss = model(inputs, labels)
            else:
                # model does not shift targets => do it our self if dataset also does not do this 
                #if "shift_targets" in model.config.keys() and not model.config.shift_targets:
                if not model.config.shift_targets:
                    outputs = outputs[..., :-1, :].contiguous()
                    labels = labels[..., 1:].contiguous()
                warn_once(log, f'Loss function outside of model (e.g. for pretrained models) is not fixed yet')
                outputs, _ = model(inputs)
                #log.critical(outputs)
                outputs = F.log_softmax(outputs, dim=2)
                loss = F.nll_loss(outputs.view(-1, outputs.size(-1)),labels.view(-1), ignore_index=-1)
            loss /= gradient_accumulation_steps #make all mini-batches account as one large batch
            loss.backward()
        #clip gradients to prevent vanishing/exploding gradient problem
        # (https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
        if isinstance(cfg.run.get("grad_clip"), float) and cfg.run.grad_clip > 0.0:
            #nn.utils.clip_grad_value_(model.parameters(), clip_value=cfg.run.grad_clip)
            #karpathy uses norm gradient clipping which scales the pre-existing gradients with the grad_clip value
            nn.utils.clip_grad_norm_(model.parameters(), cfg.run.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)  # Zero gradients after gradient accumulation
        running_loss += loss.item() * gradient_accumulation_steps 
        #log loss
        if i % cfg.run.log_steps_interval == 0:
            last_loss = running_loss / cfg.run.log_steps_interval # loss per batch
            log.info(f'  batch {i} step {i * cfg.dataset.dataloader.batch_size} loss: {last_loss}. time: {(time() - train_batch_time)*1000:.2f}ms')
            
            # wandb log only for train loss
            wandb_log({"train/loss": last_loss, "train/batch": i, "train/num_samples": i * cfg.dataset.dataloader.batch_size}) 
            train_batch_time = time()
            running_loss = 0
            ## TODO tensorboard logging and other types of reporting
        if mini_run and i>=200: # run for more than one data point
            break
        
        # eval for long epochs aka dataset with a LOT of data
        if i % cfg.run.eval_steps_interval == 0 and eval_data_loader is not None:
            losses, mean = eval_model(cfg, device, model, eval_data_loader)
            model.train()
            log.info(f'AVERAGE EVAL LOSS FOR BATCHES {i}/{run_len}: {mean.item()}')
            # log for wandb
            wandb_log_metrics = {}
            wandb_log_metrics["validate/loss"] = mean.item()
            # wandb_log_metrics["train_loss"] = last_loss
            wandb_log_metrics["train/batch"] = i
            wandb_log_metrics["train/num_samples"] = i * cfg.dataset.dataloader.batch_size
            wandb_log(wandb_log_metrics)

        if i % cfg.run.save_steps_interval == cfg.run.save_steps_interval - 1:
            ## interval or end of training, epochs is also 1 for mini_run
            # last_checkpoint is the absolute filepath of the saved checkpoint
            last_checkpoint: str = save_checkpoint(
                #cfg=cfg, 
                model=model, 
                optimizer=optimizer, 
                #dataset=cfg.dataset.name,
                timestamp=timestamp, 
                epoch=epoch, 
                metrics=loss, 
                #model_cfg=cfg.model, 
                #tokenizer_cfg=cfg.tokenizer,
                #quant_cfg = cfg.get('quantization', None),
                steps=i * cfg.dataset.dataloader.batch_size,
                )
            
        # advance learning rate
        if cfg.run.scheduler_step_type == 'steps':
            _s = cfg.run.get('scheduler_steps_interval')
            if _s is not None and _s > 0 and i % cfg.run.scheduler_steps_interval == (_s-1) and scheduler is not None:
                scheduler.step()
                new_lr = scheduler.get_last_lr()[0]
                log.info(f'New learning rate: {new_lr}')

    return last_loss, i, False  # No early stop

@torch.no_grad()
def eval_model(cfg: DictConfig, device, model: nn.Module, 
            eval_data: Union[torch_data.DataLoader, torch_data.dataloader._MultiProcessingDataLoaderIter]) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the accuracy of a model by feeding it input data during eval mode for a specified
    number of batches. The number of batches is configurable by the eval_iters field inside of the train
    hydra config.

    The function returns two Tensors:
        vlosses: The loss of every batch
        avg_loss: The average of every vloss
    """
    model.eval()
    #if isinstance(eval_data, torch_data.DataLoader):
    #    log.debug(f'Casting eval dataloader to iterable.')
    #    eval_data: torch_data.dataloader._MultiProcessingDataLoaderIter = iter(eval_data)

    if "max_iters" in  cfg.run and cfg.run.max_iters is not None and cfg.run.max_iters > 0:
        max_len = len(eval_data)
        run_len = min(max_len, cfg.run.max_iters)
        log.info(f"eval_data len is {max_len}, max_iters set to {cfg.run.max_iters}. Running eval for {run_len}")
    else:
        max_len = len(eval_data)
        run_len = len(eval_data)
        log.info(f"eval_data len is {max_len}, max_iters set to {None}. Running eval for {run_len}")

    vlosses = torch.zeros(run_len+1)
    # TODO use cfg.run.eval_iters for eval iter limits when we need it
    for i, vdata in enumerate(eval_data, 1):
        if i > run_len:
            break

        vinputs = None
        vlabels = None
        if len(vdata) > 2:
            vinputs = vdata['input_ids']
            vlabels = vdata['labels']
            vattention_mask = vdata['attention_mask']
        elif len(vdata) == 2:
            vinputs, vlabels = vdata
        else:
            log.error(f"unsupported dataloader output. len was {len(vdata)}. ")
            raise NotImplementedError

        vinputs = vinputs.to(device=device_singleton.device)
        vlabels = vlabels.to(device=device_singleton.device)
        #print(i, vinputs.size(), vlabels.size())
        #print(vinputs, vlabels)
        if cfg.model.calc_loss_in_model:
            voutputs, vloss = model(vinputs, vlabels)
            #print(voutputs)
            #print(f" loss {vloss}")
         #   log.debug(f" loss {vloss}")
        else:
            # this is wrong!
            voutputs, _ = model(vinputs)
            voutputs_softmax = torch.nn.functional.log_softmax(voutputs, dim=2)
            vloss = torch.nn.functional.nll_loss(voutputs_softmax.view(-1, voutputs_softmax.size(-1)),vlabels.view(-1), ignore_index=-1)
        #print(vloss.item())
        vlosses[i] = vloss.item()
        #print(vlosses)
    avg_vloss = vlosses.mean()
    return vlosses, avg_vloss