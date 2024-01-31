from brevitas import nn as qnn
from brevitas.graph import quantize
from torch.nn import Module, ModuleDict
import logging
from qtransform.quantization import Quantizer, ModelQuantConfig
from qtransform.classloader import get_data
from typing import Dict

#brevitas allows tweaking the quantization hyperparameters for each layer with the parameter weight_quant
#idea: pass these configs from hydra conf to each layer and override default configs found in brevitas.nn.scale_int
#the default for linear layers is for example Int8WeightPerTensorFloat (8 bits, int, per tensor, etc.)

log = logging.getLogger(__package__)

class BrevitasPTQQuantizer(Quantizer):
    def get_quantized_model(quant_cfg: ModelQuantConfig, inplace=False) -> Module:
        log.info(f'Quantizing model')
        model = quantize.preprocess_for_quantize(model)
        model = quantize.quantize(model)
        return model
    
    def train_qat(model: Module, function: any, args: list) -> Module:
        """
            Unlike pytorch, no special function has to be called in order to calibrate the qparams and train the model.
        """
        return function(model, *args)

    def export_model(model: Module, filepath: str) -> None:
        #Idea: something along the lines of export_qonnx(model, export_path=filepath)
        raise NotImplementedError