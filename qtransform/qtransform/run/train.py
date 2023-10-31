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

    """
    model = Net()
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    """
    # lets go
    quant_cfg = cfg.get('quantization')
    if quant_cfg and quant_cfg.quantize:    
        quant_cfg.device = device.type
        from qtransform.quantization import get_quantizer
        quantizer = get_quantizer(quant_cfg)
        #add qat qparams (scale and zero)
        model = quantizer.get_quantized_model(model)
        #calibrate the scales for each weight and activation
        model = quantizer.train_qat(model, train, [cfg, device, train_datalaoder, eval_dataoader, optimizer,scheduler, timestamp])
        log.debug(f'Quantized model: \n{model}')    
        output_path = os.path.join('outputs/models',f'quantized_{cfg.model.cls}_{timestamp}')
        model = quantizer.export_model(model, output_path)
    else:
        train(cfg=cfg, device=device, model=model, train_data_loader=train_datalaoder, eval_data_loader=eval_dataoader, optimizer=optimizer, scheduler=scheduler, timestamp=timestamp)


def train(model: nn.Module, cfg: DictConfig, device, train_data_loader: data.DataLoader, eval_data_loader: data.DataLoader,
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
        # save model checkpoint
        # in case we want this stuff to be configurable via env, then this should maybe be handled by hydra
        #if  __package__.split(".")[0].upper() + "_" + "model_dir".upper() in os.environ:
        #    chkpt_folder = __package__.split(".")[0].upper() + "_" + "model_dir".upper()
        #else:
        if epoch % 100 == 0:
            return
        chkpt_folder = os.path.join(os.getenv("HOME"), *__package__.split("."), "model_dir")
        if "model_dir" in cfg.run:
            if os.path.isabs(cfg.run.model_dir):
                chkpt_folder = cfg.run.model_dir
            else:
                chkpt_folder = os.path.join(hydra.core.hydra_config.HydraConfig.get().runtime.cwd, "outputs", cfg.run.model_dir)
        os.makedirs(chkpt_folder, exist_ok=True)
        if epoch % cfg.run.save_epoch_interval == 0:
            checkpoint_path = os.path.join(chkpt_folder,f'{cfg.model.cls}_{timestamp}__epoch_{epoch}')
            torch.save(obj={
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "metrics": metrics,
                }, f=checkpoint_path)
            log.info(f"Model checkpoint saved to {checkpoint_path}")
        # advance learning rate
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
        #if model.quant:
            #fake quantize inputs
        #    inputs = model.quant(inputs)
        if cfg.model.calc_loss_in_model:
            outputs, loss = model(inputs, labels)
        else:
            outputs = model(inputs)
            loss = F.nll_loss(outputs, labels)
        """TODO: pytorch support maybe"""
        #if model.dequant:
        #    outputs = model.dequant(outputs)
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
