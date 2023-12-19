from typing import Tuple, Union
import torch
import torch.nn as nn
import torch.ao.quantization as qnn #eager mode quantization api (https://pytorch.org/docs/stable/quantization-support.html#quantization-api-reference)
import logging
from qtransform.quantization import Quantizer

log = logging.getLogger(__package__)
#DeprecationWarning prevents this class from being found by qtransform.classloader.get_data()
@DeprecationWarning
class TorchQuantizer(Quantizer):
    """
        Deprecated implementation of pytorch QAT based on a hydra qconfig file. It quantizes a model in CPU.
        It is not used within this project as GPU support is not natively provided by Torch and instead done with TensorRT. 
        This results in license complications.
    """
    def __init__(self, quant_cfg: any):
        super().__init__(quant_cfg)
    #TODO: Support BackendConfig (https://pytorch.org/docs/stable/generated/torch.ao.quantization.backend_config.BackendConfig.html#torch.ao.quantization.backend_config.BackendConfig)
    def get_quantized_model(self, model: nn.Module) -> nn.Module:
        """
            !!!This function only supports CPU Quantization currently as Torch GPU Quantization requires TensorRT!!!

            Quantizes a model with the Eager Mode or FXGraph pytorch quantization framework based on a specified hydra config. 
            It appends quantization parameters (scale and zero) to either each layer or each weight within a layer 
            and updates the qparams based on a specified torch observer. 
            The function returns the quantized model overwriting the model from the argument and the functions
            needed to train the model and convert it supporting the torch backend it is supposed to train on.
            The returned model is not quantized yet, that has to be done with the supplied convert method.
        """
        #based on quant_cfg return functions for quantization
        self.convert_fn: function = qnn.convert
        self.prepare_fn: function = qnn.prepare_qat
        self.quantize_fn: function = qnn.quantize_qat
        #eager mode quantization supports cpu, fxgraph supports gpu
        if self.quant_cfg.device == 'cuda':
            log.critical(f'Pytorch GPU quantization support requires tensorrt. That is not supported by this project yet.')
            raise ValueError()
            #import torch.ao.quantization.quantize_fx as quant_fx #for gpu support
            #backend = 'TensorRT'
            #self.convert_fn = quant_fx.convert_fx
            #self.prepare_fn = quant_fx.prepare_qat_fx
        elif self.quant_cfg.device == 'cpu':
            #only x86 and ARM are supported currently
            from platform import processor
            backend = 'x86' if 'x86' in processor() else 'fbgemm'
        else:
            log.warning(f'Pytorch quantization currently only supports CPU and GPU devices, not {self.quant_cfg.device}. Unexpected behavior might happen.')
        torch.backends.quantized.engine = backend
        log.debug(f'Using backend: {torch.backends.quantized.engine} for quantization')
        model = model.train()

        #scheme
        _qscheme = 'per_' + self.quant_cfg.args.scope + '_' + self.quant_cfg.args.scheme
        try:
            qscheme =  getattr(torch, _qscheme)
        except AttributeError:
            log.warning(f'Scheme {_qscheme} not found within torch. Defaulting to per_channel_affine')
            qscheme = torch.per_channel_affine

        #only QAT is supported currently
        if self.quant_cfg.kind == 'qat':
            #fake quant, TODO: dynamically add quant to beginning of forward and dequant to end of forward
            model.quant = qnn.QuantStub()
            model.dequant = qnn.DeQuantStub() 
            dtype = TorchQuantizer.get_dtype(bits=self.quant_cfg.args.bit_width, signed=self.quant_cfg.args.signed)
            quant_max, quant_min = TorchQuantizer.get_clipping_range_dtype(
                _quant_max = self.quant_cfg.args.max_value, 
                _quant_min = self.quant_cfg.args.min_value, 
                dtype=dtype)
            activation=qnn.observer.MinMaxObserver.with_args(dtype=dtype, quant_min = quant_min, quant_max = quant_max, qscheme=qscheme)
            weight=qnn.observer.default_observer.with_args(dtype=dtype, quant_min = quant_min, quant_max = quant_max, qscheme=qscheme)
            model.qconfig = qnn.qconfig.QConfig(weight, activation)
            #adds qparams to the model directly without copying it for memory usage reasons
            try:
                model_qat = self.prepare_fn(model, inplace=False)
            except torch.fx.proxy.TraceError:
                log.error(f'Model cannot be quantized without changing its structure due Torch\'s FXGraph Quant API requiring models to be symbolically traceable.')
                #TODO: Change structure of model
                raise ValueError()
        else:
            log.critical(f'No quantization options other than QAT are supported as of currently.')
            raise ValueError()
        #depending on quant_config return the next steps for quantization
        return model_qat
    
    def train_qat(self, model: nn.Module, function: any, args: list) -> nn.Module:   
        log.info(f'Performing QAT with torch quantization framework.')
        return self.quantize_fn(model, function, args, inplace = False)
    
    def export_model(self, model: nn.Module, filepath: str) -> None:
        #actually quantize the model by applying the qparams to the corresponding weights
        #if present, the (de)quant stubs are replaced with (de)quantize operations respectively
        model = self.convert_fn(model.eval())
        torch.save(model, filepath)
        log.info(f'Quantized model saved in \"{filepath}\"')

    def get_dtype(bits: int, signed: bool) -> torch.dtype:
        """
            Constructs the torch.dtype class from the hydra config based on the bit length and
            whether the dtype is supposed to capsule negative values or not. It will default
            to qint8 if it the dtype is not found within torch, currently only int representations
            are supported by torch quantization (except for qfloat16 which is omited in this project).
        """
        _dtype = 'q' + 'u' if signed else '' + 'int' + bits
        try:
            dtype: torch.dtype = getattr(torch, _dtype)
        except AttributeError:
            log.warning(f'Pytorch does not have dtype: {_dtype}. Defaulting to qint8')
            dtype: torch.dtype = torch.qint8
        return dtype

    def get_clipping_range_dtype(_quant_max: Union[int, None], _quant_min: Union[int, None], dtype: torch.dtype) -> Tuple[int, int]:
        """
            Infer the clipping range for the pytorch observers based on either the hydra config (if present)
            or the underlying dtype used during quantization. It returns a tuple with the maximum and minimum
            value.
        """
        log.critical(f'quant_max: {_quant_max} {type(_quant_max)}')
        if not _quant_max:
            quant_max = torch.iinfo(dtype).max
        if not _quant_min:
            quant_min = torch.iinfo(dtype).min
        return (quant_max, quant_min)