import logging
from typing import Any, Tuple
from omegaconf import DictConfig, OmegaConf, open_dict
import os
import torch
from torch import nn
from torch import optim
from torch.optim import lr_scheduler
from torch.utils import data
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import torch.nn.functional as F
from qtransform.utils import load_checkpoint, save_checkpoint
from pprint import PrettyPrinter
from qtransform import device_singleton

log = logging.getLogger(__name__)

def run(cfg: DictConfig):
    """ launches training with provided config"""
    log.info("================")
    log.info("Running Training")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")
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

    from qtransform.dataset import get_data, get_loader, DatasetWrapper
    data_wrapper: DatasetWrapper = get_data(cfg.dataset)
    data_wrapper.load_dataset()
    dataset_train = data_wrapper.dataset_info.train
    dataset_eval = data_wrapper.dataset_info.eval
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
        log.warning(f'Vocab size of model is larger than the tokenizer vocab. Setting vocab_size to: {max_token_value} to prevent errors during inference')
        OmegaConf.update(cfg, "model.args.vocab_size", max_token_value, force_add=True)

    from qtransform.model import get_model
    model = get_model(cfg.model)
    model.train()
    #only parameters (type torch.nn.parameter.Parameter) are moved to the device, not non-named Tensors
    #this is a problem if a layer uses a non-named Tensor during the forward pass
    model.to(device=device)

    from qtransform.optim import get_optim#, get_scheduler
    log.debug(f"optim config: {cfg.optim}")
    #optimizer = optim.Adadelta(model.parameters(), lr=cfg.optim.learning_rate)
    optimizer = get_optim(model=model, optim_cfg=cfg.optim)
    log.debug(f'Configured optimizer ({type(optimizer)}): {optimizer}')
    # TODO dynamic scheduler
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1)

    last_checkpoint = None
    # lets go
    quant_cfg = cfg.get('quantization')
    if quant_cfg and quant_cfg.quantize:    
        log.debug(f'Running quantized model')
        from qtransform.quantization import get_quantizer
        quantizer, model_quant_cfg = get_quantizer(quant_cfg, model=model)
        model, replace_layers_later = quantizer.get_quantized_model(model_quant_cfg, inplace=True)
        # TODO make this a decorator so it can return stuff
        last_checkpoint = quantizer.train_qat(model, train, [cfg, device, train_dataloader, eval_dataloader, optimizer,scheduler, timestamp])
        #quantize last layers (batchnorm). parmams last saved checkpoint do not entirely reflect current model anymore 
        if replace_layers_later is not None:
            model = quantizer.get_quantized_model(replace_layers_later)
    else:
        #if hasattr(log,"trace"): log.trace(model)
        last_checkpoint = train(cfg=cfg, device=device, model=model, train_data_loader=train_dataloader, eval_data_loader=eval_dataloader, optimizer=optimizer, scheduler=scheduler, timestamp=timestamp)
    # maybe subsequent jobs can be managed by hydra in the future?
    # when this paradigm comes up more frequently we have to make this a thing ....
    log.debug("Finished training model")
    #write checkpoint into fifo if model is not exported, otherwise write path to onnx model into fifo
    from qtransform.utils.helper import write_to_pipe
    if cfg.run.get("export") and last_checkpoint:
        from qtransform.run import export
        from hydra import compose
        #load another entire hydra config with run=export, then override the current run config with export
        #this saves having to re-initialize the globalhydra configuration and further redundant config steps
        #(https://hydra.cc/docs/advanced/compose_api/ and https://github.com/facebookresearch/hydra/issues/440)
        export_cfg = compose(config_name="config", overrides=["run=export"])
        with open_dict(cfg):
            cfg.run = export_cfg.run
        OmegaConf.update(cfg, "run.from_checkpoint", last_checkpoint, force_add=True)
        OmegaConf.update(cfg, "run.running_model", True, force_add=True)
        if quant_cfg and quant_cfg.quantize:
            OmegaConf.update(cfg, "run.export_fn", "qonnx", force_add=True)
        else:
            OmegaConf.update(cfg, "run.export_fn", "onnx", force_add=True)
        kwargs = {"model": model}
        export.run(cfg, **kwargs)
    else:
        #write checkpoint into fifo
        write_to_pipe(cfg, last_checkpoint)
        
def train(model: nn.Module, cfg: DictConfig, device, train_data_loader: data.DataLoader, eval_data_loader: data.DataLoader,
           optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler, timestamp: datetime) -> Any:
    """ training over epochs with periodic logging and saving"""
    #print(model)
    mini_run = False
    epochs_to_run = None
    last_checkpoint = None
    if cfg.run.epochs == 0:
        cfg.run["epochs"] = 1
        log.warn("cfg.run.epochs is 0, performing mini training dry run")
        mini_run = True

    if "from_checkpoint" in cfg.run and cfg.run.from_checkpoint:
        log.info(f"Resuming training from {cfg.run.from_checkpoint}")
        from_epoch, checkpoint = load_checkpoint(cfg)
        log.info(f"Epoch is {from_epoch}, running for {cfg.run.epochs}")
        cfg.run.epochs = from_epoch + cfg.run.epochs
  
        if 'model_state_dict' not in checkpoint:
            log.error("Can not load checkpoint with no model_state_dict")
            raise KeyError
        if 'optimizer_state_dict' not in checkpoint:
            log.error("Can not load checkpoint with no optimizer_state_dict")
            raise KeyError
        if 'quantized' not in checkpoint:
            log.warning(f'No info specified if checkpoint is quantized. Assuming false.')
        elif checkpoint["quantized"]:
            #skip qparams from checkpoint
            #for some reason, mlp qparams are saved within checkpoint but not the ones from mha
            #TODO: investigate
            from brevitas import config
            config.IGNORE_MISSING_KEYS = True
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        metrics = {}
        
        if 'metrics' not in checkpoint:
            log.warn("no metrics found in checkpoint")
        else:
            metrics = checkpoint['metrics']

        epochs_to_run = range(from_epoch + 1, cfg.run.epochs + 1)
    else:
        log.info(f"Starting new training")
        epochs_to_run = range(1, cfg.run.epochs + 1)

        
    # training loop
    for epoch in epochs_to_run:
        log.info(f"EPOCH: {epoch}/{cfg.run.epochs}")

        metrics = train_one_epoch(cfg, device, model, train_data_loader, optimizer, mini_run)

        ## eval
        if epoch % cfg.run.eval_epoch_interval == 0 and eval_data_loader is not None:
            losses, mean = eval_model(cfg, device, model, eval_data_loader)
            log.info(f'AVERAGE EVAL LOSS FOR EPOCH {epoch}/{cfg.run.epochs}: {mean.item()}')
        log.info(str(metrics))

        if epoch % cfg.run.save_epoch_interval == 0 or epoch % cfg.run.epochs == 0: 
            ## interval or end of training, epochs is also 1 for mini_run
            # last_checkpoint is the absolute filepath of the saved checkpoint
            last_checkpoint: str = save_checkpoint(cfg=cfg, 
                model=model, 
                optimizer=optimizer, 
                timestamp=timestamp, 
                epoch=epoch, 
                metrics=metrics, 
                model_cfg=cfg.model, 
                tokenizer_cfg=cfg.dataset.tokenizer,
                quant_cfg = cfg.get('quantization', None))

        # advance learning rate
        scheduler.step()
    return last_checkpoint

def train_one_epoch(cfg: DictConfig, device, model: nn.Module, train_data: data.DataLoader,
           optimizer: optim.Optimizer, mini_run: bool=False) -> Any:
    """ training loop over steps/batches """
    model.train() #if it was quantized, it could have been set to eval
    last_loss = 0
    running_loss = 0
    #cfg is entire hydra config
    for i, data in enumerate(train_data):
        optimizer.zero_grad()  # Zero your gradients for every batch
        #token tensor of length block_size (context length)
        inputs, labels = data
        inputs = inputs.to(device_singleton.device)
        labels = labels.to(device_singleton.device)
        if cfg.model.calc_loss_in_model:
            outputs, loss = model(inputs, labels)
        else:
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
        loss.backward()
        #clip gradients to prevent vanishing/exploding gradient problem
        # (https://neptune.ai/blog/understanding-gradient-clipping-and-how-it-can-fix-exploding-gradients-problem)
        if isinstance(cfg.run.get("grad_clip"), float) and cfg.run.grad_clip > 0.0:
            nn.utils.clip_grad_value_(model.parameters(), clip_value=cfg.run.grad_clip)
        optimizer.step()

        running_loss += loss.item()
        if i % cfg.run.log_steps_interval == 0:
            last_loss = running_loss / cfg.run.log_steps_interval # loss per batch
            log.info(f'  batch {i} loss: {last_loss}')
            running_loss = 0
            ## TODO tensorboard logging and other types of reporting
        if mini_run and i>=200: # run for more than one data point
            break
        #dataloaders iterate through entire dataset
        #problematic if datasets are large (openwebtext ~21GB) and testing should be done
        elif i>= cfg.run.max_iters:
            break
    return last_loss    

@torch.no_grad()
def eval_model(cfg: DictConfig, device, model: nn.Module, evaldata: data.DataLoader) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Evaluates the accuracy of a model by feeding it input data during eval mode for a specified
    number of batches. The number of batches is configurable by the eval_iters field inside of the train
    hydra config.

    The function returns two Tensors:
        vlosses: The loss of every batch
        avg_loss: The average of every vloss
    """
    model.eval()
    vlosses = torch.zeros(cfg.run.eval_iters)
    i = 0
    while i < cfg.run.eval_iters:
        vdata = next(iter(evaldata))
        vinputs, vlabels = vdata
        if cfg.model.calc_loss_in_model:
            voutputs, vloss = model(vinputs, vlabels)
        else:
            voutputs = model(vinputs)
            vloss = F.nll_loss(voutputs, vlabels)
        vlosses[i] = vloss.item()
        i += 1
    avg_vloss = vlosses.mean()
    model.train()
    return vlosses, avg_vloss