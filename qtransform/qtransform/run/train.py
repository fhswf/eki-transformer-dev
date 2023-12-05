import logging
from typing import Any
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
    device_singleton.device = cfg.device
    device = device_singleton.device
    if device.type == 'cuda':
        cuda_kwargs = {'pin_memory': True,}
        #struct flag of dictconf prevents additional keys to be added (https://omegaconf.readthedocs.io/en/latest/usage.html#struct-flag)
        with open_dict(cfg.dataset.dataloader):
            cfg.dataset.dataloader.update(cuda_kwargs)
        import sys
        log.critical(cfg.dataset.dataloader)
        sys.exit(1000)
    torch.manual_seed(cfg.seed)    
    log.info(f"number of torch dataloader: {str(cfg.dataset.dataloader.num_workers)}")

    from qtransform.model import get_model
    model = get_model(cfg.model)
    model.train()
    #only parameters (type torch.nn.parameter.Parameter) are moved to the device, not Tensors
    #this is a problem if a layer uses a Tensor during the forward pass that is not moved to the device inside of the class
    model.to(device)

    from qtransform.dataset import get_data, get_loader
    data_lodaer_maybe_tuple = get_data(cfg.dataset)
    if isinstance(data_lodaer_maybe_tuple, tuple):
        train_datalaoder, eval_dataoader = data_lodaer_maybe_tuple
    else:
        train_datalaoder = data_lodaer_maybe_tuple
        eval_dataoader = None
    
    if not isinstance(train_datalaoder, data.DataLoader):
        train_datalaoder = get_loader(data=train_datalaoder, dataloader_cfg=cfg.dataset.dataloader)
    if eval_dataoader is not None and not isinstance(eval_dataoader, data.DataLoader):
        eval_dataoader   = get_loader(data=eval_dataoader, dataloader_cfg=cfg.dataset.dataloader)

    # TODO dynamic optim
    # from qtransform.optim import get_optim, get_scheduler
    log.debug(f"optim config: {cfg.optim}")
    optimizer = optim.Adadelta(model.parameters(), lr=cfg.optim.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1)

    last_checkpoint = None
    # lets go
    quant_cfg = cfg.get('quantization')
    if quant_cfg and quant_cfg.quantize:    
        log.debug(f'Running quantized model')    
        quant_cfg.device = device.type
        from qtransform.quantization import get_quantizer
        quantizer, model_quant_cfg = get_quantizer(quant_cfg, model=model)
        #add qat qparams (scale and zero)
        model = quantizer.get_quantized_model(model_quant_cfg, inplace=True)
        #calibrate the scales for each weight and activation
        # TODO make this a decorater so it can return stuff
        model = quantizer.train_qat(model, train, [cfg, device, train_datalaoder, eval_dataoader, optimizer,scheduler, timestamp])
    else:
        last_checkpoint = train(cfg=cfg, device=device, model=model, train_data_loader=train_datalaoder, eval_data_loader=eval_dataoader, optimizer=optimizer, scheduler=scheduler, timestamp=timestamp)

    # maybe subsequent jobs can be managed by hydra in the future?
    # when this paradigm comes up more frequently we have to make this a thing ....
    if cfg.run.get("export") and last_checkpoint:
        from qtransform.run import export
        OmegaConf.update(cfg, "run.from_checkpoint", last_checkpoint, force_add=True)
        export.run(cfg)
        
def train(model: nn.Module, cfg: DictConfig, device, train_data_loader: data.DataLoader, eval_data_loader: data.DataLoader,
           optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler, timestamp: datetime) -> Any:
    """ training over epochs with periodic logging and saving"""
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
        log.info(str(metrics))

        ## eval
        #if epoch % cfg.run.eval_epoch_interval == 0 and eval_data_loader is not None:
        #    eval_result = eval_model(cfg, device, model, eval_data)
        #    # TODO log data

        if epoch % cfg.run.save_epoch_interval == 0 or epoch % cfg.run.epochs == 0: 
            ## interval or end of training, epochs is also 1 for mini_run
            last_checkpoint = save_checkpoint(cfg=cfg, model=model, optimizer=optimizer, timestamp=timestamp, epoch=epoch, metrics=metrics, model_cfg=cfg.model)

        # advance learning rate
        scheduler.step()
    return last_checkpoint

def train_one_epoch(cfg: DictConfig, device, model: nn.Module, train_data: data.DataLoader,
           optimizer: optim.Optimizer, mini_run: bool=False) -> Any:
    """ training loop over steps/batches """
    model.train() #if it was quantized, it could have been set to eval
    last_loss = 0
    running_loss = 0
    for i, data in enumerate(train_data):
        optimizer.zero_grad()  # Zero your gradients for every batch
        inputs, labels = data
        inputs = inputs.to(device_singleton.device)
        labels = labels.to(device_singleton.device)
        if cfg.model.calc_loss_in_model:
            outputs, loss = model(inputs, labels)
        else:
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
        """TODO: pytorch support maybe"""
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % cfg.run.log_steps_interval == 0:
            last_loss = running_loss / cfg.run.log_steps_interval # loss per batch
            log.info(f'  batch {i} loss: {last_loss}')
            running_loss = 0
            ## TODO tensorboard logging and other types of reporting
        if mini_run and i>=200: # run for more than one data point
            break
    return last_loss    


# TODO

@torch.no_grad()
def eval_model(cfg: DictConfig, device, model: nn.Module, evaldata: data.Dataset):
    for i, vdata in enumerate(evaldata.loader):
        vinputs, vlabels = vdata
        voutputs = model(vinputs)
        vloss = loss_fn(voutputs, vlabels)
        running_vloss += vloss
    avg_vloss = running_vloss / (i + 1)
    return avg_loss, avg_vloss


@torch.no_grad()
def estimate_loss(cfg: DictConfig, model: nn.Module):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            with ctx:
                logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out
