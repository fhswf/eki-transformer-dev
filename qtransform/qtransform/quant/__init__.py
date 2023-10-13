import torch
import torch.nn as nn
import torch.ao.quantization as quant
from omegaconf import DictConfig
import logging
from dataclasses import dataclass

log = logging.getLogger(__name__)
#todo: differentiate if model is running on cpu or gpu
#and set backend according to the architecture of the device
backend = 'x86'


@dataclass
class QuantArgs:
    dtype: str
    observer: str
    scheme: str
    scope: str

@dataclass
class QuantConfig():
    kind: str
    args: QuantArgs

def get_quantized_model(model: nn.Module, quant_cfg: DictConfig):
    log.debug(f'Quantizing with parameters: {quant_cfg}')
    torch.backends.quantized.engine = backend
    log.debug(f'Using backend: {torch.backends.quantized.engine} for quantization')
    q_cfg = QuantConfig(**quant_cfg)
    if q_cfg.kind == 'qat':
        #fake quant
        model.quant = quant.QuantStub()
        model.dequant = quant.DeQuantStub() 
        #qconfig
        try:
            dtype: torch.dtype = getattr(torch, q_cfg.args.dtype)
        except AttributeError:
            log.warning(f'Cannot quantize model with dtype: {quant_cfg.dtype}. Defaulting to qint8')
            dtype: torch.dtype = torch.qint8
        quant_max = torch.iinfo(dtype).max
        quant_min = torch.iinfo(dtype).min
        #TODO: make observers and scheme generic
        activation=quant.observer.MinMaxObserver.with_args(dtype=dtype, quant_min = quant_min, quant_max = quant_max)
        weight=quant.observer.default_observer.with_args(dtype=dtype, quant_min = quant_min, quant_max = quant_max)
        model.qconfig = quant.qconfig.QConfig(weight, activation)
        #adds qparams to the model directly without copying it for memory usage reasons
        model_qat = torch.quantization.prepare_qat(model, inplace=True)
    else:
        log.critical(f'No quantization options other than QAT are supported as of currently.')
        raise ValueError()
    
    return model