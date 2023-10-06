import logging
from typing import Any
from omegaconf import DictConfig
import hydra
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

    cuda = None
    device = None
    if "cuda" in cfg:
        cuda = cfg.cuda and torch.cuda.is_available()
    else:
        cuda = torch.cuda.is_available()
    mps = None
    if "mps" in cfg:
        mps = cfg.mps and torch.backends.mps.is_available()
    else:
        mps = torch.backends.mps.is_available()

    torch.manual_seed(cfg.seed)    
    if cuda:
        device = torch.device("cuda")
        cuda_kwargs = {'pin_memory': True,}
        cfg.dataset.dataloader.update(cuda_kwargs)
    elif mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    log.info(f"using device: {str(device)}")
    log.info(f"number of torch dataloader: {str(cfg.dataset.dataloader.num_workers)}")

    from qtransform.model import get_model
    model = get_model(cfg.model)
    model.train()
    model.to(device)

    from qtransform.dataset import get_data, get_loader
    train_data, eval_data = get_data(cfg.dataset)
    train_datalaoder = get_loader(data=train_data, dataloader_cfg=cfg.dataset.dataloader)
    eval_dataoader   = get_loader(data=eval_data, dataloader_cfg=cfg.dataset.dataloader)

    # from qtransform.optim import get_optim, get_scheduler
    log.debug(f"optim config: {cfg.optim}")
    optimizer = optim.Adadelta(model.parameters(), lr=cfg.optim.learning_rate)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=1)

    train(cfg=cfg, device=device, model=model, train_data_loader=train_datalaoder, eval_data_loader=eval_dataoader, optimizer=optimizer, scheduler=scheduler, timestamp=timestamp)


def train(cfg: DictConfig, device, model: nn.Module, train_data_loader: data.DataLoader, eval_data_loader: data.DataLoader,
           optimizer: optim.Optimizer, scheduler: lr_scheduler.LRScheduler, timestamp: datetime) -> Any:
    """ training over epochs with periodic logging and saving"""
    epochs_to_run = None
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
        metrics = train_one_epoch(cfg, device, model, train_data_loader, optimizer)
        log.info(str(metrics))

        ## eval
        #if epoch % cfg.run.eval_epoch_interval == 0:
        #    eval_result = eval_model(cfg, device, model, eval_data)
        #    # TODO log data
        if epoch % cfg.run.save_epoch_interval == 0:
            save_checkpoint(cfg, model, optimizer, timestamp, metrics, epoch)
    
        scheduler.step()


def train_one_epoch(cfg: DictConfig, device, model: nn.Module, train_data: data.DataLoader,
           optimizer: optim.Optimizer) -> Any:
    """ training loop over steps/batches """
    # TODO comute more metrics
    last_loss = 0
    running_loss = 0
    for i, data in enumerate(train_data):
        optimizer.zero_grad()  # Zero your gradients for every batch
        # TODO 
        #data.to(device)
        inputs, labels = data

        outputs = model(inputs)
        loss = F.nll_loss(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if i % cfg.run.log_steps_interval == 0:
            last_loss = running_loss / cfg.run.log_steps_interval # loss per batch
            log.info(f'  batch {i+1} loss: {last_loss}')
            running_loss = 0
            ## TODO tensorboard logging and other types of reporting

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
