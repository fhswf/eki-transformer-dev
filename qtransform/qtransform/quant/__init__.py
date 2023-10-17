import torch
import torch.nn as nn
import torch.ao.quantization as quant #eager mode quantization api (https://pytorch.org/docs/stable/quantization-support.html#quantization-api-reference)
from omegaconf import DictConfig
import logging
from dataclasses import dataclass
from qtransform.utils.introspection import get_classes
from typing import Tuple


log = logging.getLogger(__name__)
#dtypes = [torch.qint8, torch.quint8, torch.quint32]

@dataclass
class QuantArgs:
    dtype: str
    observer: str
    scheme: str
    scope: str

@dataclass
class QuantConfig():
    quantize: bool
    kind: str
    device: str
    args: QuantArgs

#TODO: Support BackendConfig (https://pytorch.org/docs/stable/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig)
def get_quantized_model(model: nn.Module, quant_cfg: DictConfig) -> Tuple[nn.Module, any, any]:
    """
        Quantizes a model with the Eager Mode or FXGraph pytorch quantization framework based on a specified hydra config. 
        It appends quantization parameters (scale and zero) to either each layer or each weight within a layer 
        and updates the qparams based on a specified torch observer. 
        The function returns the quantized model overwriting the model from the argument and the functions
        needed to train the model and convert it supporting the torch backend it is supposed to train on.
        The returned model is not quantized yet, that has to be done with the supplied convert method.
    """
    log.debug(f'Quantizing with parameters: {quant_cfg}')
    #based on quant_cfg return functions for quantization
    convert_fn: function = quant.convert
    prepare_fn: function = quant.prepare_qat
    quantize_fn: function = quant.quantize_qat

    q_cfg = QuantConfig(**quant_cfg)
    #eager mode quantization supports cpu, fxgraph supports gpu
    if q_cfg.device == 'cuda':
        log.critical(f'Pytorch GPU quantization support requires tensorrt. That is not supported by this project yet.')
        raise ValueError()
        #import torch.ao.quantization.quantize_fx as quant_fx #for gpu support
        #backend = 'TensorRT'
        #convert_fn = quant_fx.convert_fx
        #prepare_fn = quant_fx.prepare_qat_fx
    elif q_cfg.device == 'cpu':
        #only x86 and ARM are supported currently
        from platform import processor
        backend = 'x86' if 'x86' in processor() else 'qnnpack'
    else:
        log.warning(f'Pytorch quantization currently only supports CPU and GPU devices, not {q_cfg.device}. Unexpected behavior might happen.')
    torch.backends.quantized.engine = backend
    log.debug(f'Using backend: {torch.backends.quantized.engine} for quantization')
    model = model.train()

    #scheme
    _qscheme = 'per_' + q_cfg.args.scope + '_' + q_cfg.args.scheme
    try:
        qscheme =  getattr(torch, _qscheme)
    except AttributeError:
        log.warning(f'Scheme {_qscheme} not found within torch. Defaulting to per_channel_affine')
        qscheme = torch.per_channel_affine

    #only QAT is supported currently
    if q_cfg.kind == 'qat':
        #fake quant
        model.quant = quant.QuantStub()
        model.dequant = quant.DeQuantStub() 
        try:
            dtype: torch.dtype = getattr(torch, q_cfg.args.dtype)
        except AttributeError:
            log.warning(f'Cannot quantize model with dtype: {quant_cfg.dtype}. Defaulting to qint8')
            dtype: torch.dtype = torch.qint8
        quant_max = torch.iinfo(dtype).max
        quant_min = torch.iinfo(dtype).min
        activation=quant.observer.MinMaxObserver.with_args(dtype=dtype, quant_min = quant_min, quant_max = quant_max, qscheme=qscheme)
        weight=quant.observer.default_observer.with_args(dtype=dtype, quant_min = quant_min, quant_max = quant_max, qscheme=qscheme)
        model.qconfig = quant.qconfig.QConfig(weight, activation)
        #adds qparams to the model directly without copying it for memory usage reasons
        try:
            model_qat = prepare_fn(model, inplace=True)
        except torch.fx.proxy.TraceError:
            log.error(f'Model cannot be quantized without changing its structure due Torch\'s FXGraph Quant API requiring them to be symbolically traceable.')
            #TODO: Change structure of model
            raise ValueError()
    else:
        log.critical(f'No quantization options other than QAT are supported as of currently.')
        raise ValueError()
    #depending on quant_config return the next steps for quantization
    return model_qat, convert_fn, quantize_fn