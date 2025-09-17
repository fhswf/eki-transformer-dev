import os
import re
import time
import numpy as np
from typing import Any
import omegaconf
from omegaconf import DictConfig, open_dict, OmegaConf
from qtransform.classloader import get_data
from torch import nn, Tensor, from_numpy
from torch import compile as torch_compile
from torch import __version__ as torch_version
import logging

from dataclasses import dataclass, fields
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
# maybe only do this when it is required, for this howiever is always the case
from onnx.shape_inference import infer_shapes
from enum import IntEnum
from qtransform.utils.helper import load_checkpoint, save_checkpoint, load_onnx_model, load_state_dict_proxy, FromFile
from typing import Union, Tuple
from abc import ABC, abstractmethod
import transformers
import onnxruntime
from qtransform.quantization import get_quantizer, ModelQuantConfig, Quantizer
from qtransform import ConfigSingleton

from functools import lru_cache

import torch

# Keep track of 10 different messages and then warn again
@lru_cache(1)
def warn_once(logger: logging.Logger, msg: str):
    logger.warning(msg)

log = logging.getLogger(__name__)

class ModelType(IntEnum):
    ONNX = 0
    CHECKPOINT = 1
    PRETRAINED = 2

@dataclass
class ModelSupport():
    "Static information about capabilities of a model class"
    training: bool= False

class ModelInfoMixin():
    support = ModelSupport()
    def __init__(self):
        pass

    @classmethod
    def supports(cls) -> ModelSupport():
        return cls.support

def get_hf_pretrained(model_cfg: DictConfig) -> nn.Module:
    """
    Loads a pretrained huggingface model from model config. 
    Fields model_id and cls should exist.

    model_id: name of pretrained model (gpt2)
    cls: Base class of Model (GPT2LMHead)
    """
    model_id = model_cfg.get('model_id')
    model_cls = getattr(transformers, model_cfg.get('cls'), None)
    if model_cls is None:
        log.error(f'Could not find base model class for hf pretrained model with cfg: {model_cfg}')
        raise KeyError()
    model = model_cls.from_pretrained(model_id)
    return model

@dataclass
class ModelArgs():
    block_size: int = 1024
    vocab_size: int = 50304 # GPT-2 vocab_size of 50257, padded up to nearest multiple of 64 for efficiency
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    flash: bool = False # cuda flas hattention
    transformer_active_func: str = 'ReLU' #specify which activation function to use in MLP (feed forwad neural network)
    norm_layer: str = 'BatchNorm' # note that this is a name for a adapter module in this repository und model.modules
    single_output: bool = False # use mini runtime optimization to only predict last token, saven on some runtime but poentially currupts onnx export
    use_weight_tying: bool = True # same weights for input emb and outputi proj https://paperswithcode.com/method/weight-tying
    custom_ln: bool = False #use CustomBatchNorm1d before BatchNorm
    use_causal: bool = False
    shift_targets: bool = False # True: labels are shifted by one to the right inside the model, False: shifting is done by dataloader
    pos_layer: str = 'learned'

@dataclass
class ModelConfig():
    type: str #ONNX or DYNAMIC_CHECKPOINT
    cls: str
    calc_loss_in_model: bool
    args: ModelArgs
    from_file: FromFile = None #torch checkpoint or ONNX model path
    model_name: str = "Missing-Model-Name" # used to name the saved checkpoints
    cstr: str = None # infer model args from a config string: Mgpt2-s256-t2048-l2-h4-e512-AReLU-NBatchNormTranspose-Plearned

    def __setattr__(self, name, value):
        if name == "args" and not isinstance(value, ModelArgs):
            value = ModelArgs(**value)
        self.__dict__[name] = value

class GenericModel(nn.Module, ABC):
    """
    Generic model class to be used for transformer implementations during runs.
    It is expected that model architecture (mainly context length and embedding dimension) must be passed 
    during construction.
    """
    def __init__(self, config: ModelArgs):
        super(GenericModel, self).__init__()
        try:
            self.config = config = config if isinstance(config, ModelArgs) else ModelArgs(**config)
            log.debug(f'Applied config: \n{self.config}')
        except:   
            log.error(f'Model config \n{config}\n could not be applied. Config can only have options: {[x.name for x in fields(ModelArgs)]}')
        assert config.vocab_size is not None
        assert config.block_size is not None
        log.info(f"Model config: {self.config}")
    
    @abstractmethod
    def forward(self, idx: Tensor, targets: Tensor = None):
        raise NotImplementedError()

class QTRModelWrapper(ABC):
    """QTRModelWrapper instead of ModelWrapper to avoid naming conflicts with ONNX models"""
    #TODO: properties
    model: Union[ModelWrapper, GenericModel] 
    model_type: ModelType
    _model_cfg: ModelConfig #missing args from config are replaced with default values of dataclass
    _tokenizer_cfg: Any # TODO Any to some new data class

    def __init__(self, model_cfg: Union[ModelConfig, DictConfig]):
        pass

    @property
    def model_cfg(self):
        return self._model_cfg

    @property
    def tokenizer_cfg(self):
        return self._tokenizer_cfg

    @tokenizer_cfg.setter
    def tokenizer_cfg(self, value):
        self._tokenizer_cfg = value

    @model_cfg.setter
    def model_cfg(self, value: Union[ModelConfig, DictConfig]):
        if isinstance(value, ModelConfig):
            self._model_cfg = value
        else:
            try:
                self._model_cfg = ModelConfig(**value)
            # TODO this is temporary and should be deleted once we create new train sweeps on pc2
            except omegaconf.errors.InterpolationKeyError as e:
                log.warning(f"InterpolationKeyError in Hydra conf {e}")
                if e.full_key == "model_name":
                    OmegaConf.update(value, e.full_key, "unkown-s${.args.block_size}-t${.args.vocab_size}-l${.args.n_layer}-h${.args.n_head}-e${.args.n_embd}-A${.args.transformer_active_func}-N${.args.norm_layer}-P${.args.pos_layer}")
                    self._model_cfg = ModelConfig(**value)
                elif e.key == "model_name":
                    OmegaConf.update(value, e.key, "unkown-s${.args.block_size}-t${.args.vocab_size}-l${.args.n_layer}-h${.args.n_head}-e${.args.n_embd}-A${.args.transformer_active_func}-N${.args.norm_layer}-P${.args.pos_layer}")
                    self._model_cfg = ModelConfig(**value)
                else:
                    raise e

    @abstractmethod
    def load_model(self, model_cfg: Union[ModelConfig, DictConfig]):
        pass

    @abstractmethod
    #TODO: inference needs model cfg even though it only needs pathname to model
    #      ONNX models need model property to get config though
    def from_file(self, from_file: FromFile):
        raise NotImplementedError

    def __call__(self, idx_cond: Tensor, labels = None):
        return self.forward(idx_cond, labels)
    
    @abstractmethod
    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        raise NotImplementedError

    @abstractmethod
    def save_model(self):
        raise NotImplementedError

    #mainly to avoid conflicts with onnx models
    #no need to return model as wrapper is used
    def to(self, *args, **kwargs) -> None:
        self.model.to(*args, **kwargs)


#TODO: should the modelwrappers be stored in a singleton?
#      since one wrapper contains possibly a large model, it would not make sense to train multiple models
#      in one run
class DynamicCheckpointQTRModelWrapper(QTRModelWrapper):

    model: GenericModel
    #only if ptq/ qat is performed
    replace_layers_later: ModelQuantConfig
    quantized: bool
    quant_cfg: ModelQuantConfig
    quantizer: Quantizer
    optimizer_state_dict: DictConfig #when loading checkpoints, get optimizer state
    epochs: int #if loading checkpoint, remember how many epochs have elapsed
    metrics: dict

    def __init__(self, model_cfg):
        self.model_type = ModelType.CHECKPOINT
        self.quantized = False
        self.replace_layers_later = None
        self.quant_cfg = None
        self.optimizer_state_dict = None
        self.epochs = 0
        self.metrics = {}

        from_file: FromFile = model_cfg.get("from_file", None)
        if from_file.filename is not None:
            self.from_file(from_file)
            #block size might not be specified, necessary for dataset retrieval
            #TODO: put this in run __init__
            OmegaConf.update(model_cfg, 'cls', self.model_cfg.cls, force_add=True)
            for field in fields(self.model_cfg.args):
                OmegaConf.update(model_cfg.args,field.name, getattr(self.model_cfg.args, field.name), force_add=True)
            log.info(f'Updated model config with checkpoint parameters')
        else:
            #not that clean, problem is that checkpoint needs to be loaded in order to have model_cfg
            self.load_model(model_cfg)

    def from_file(self, from_file: FromFile):
        """
        Instantiates a torch module, initializes its state dict from a checkpoint and performs quantization
        if the checkpoint was quantized.
        """
        self.epochs, checkpoint = load_checkpoint(from_file)
        if 'model_state_dict' not in checkpoint:
            log.error("Can not load checkpoint with no model_state_dict")
            raise KeyError
        if 'optimizer_state_dict' not in checkpoint:
            log.error("Can not load checkpoint with no optimizer_state_dict")
            raise KeyError

        model_cfg = checkpoint["model_cfg"]
        self.tokenizer_cfg = checkpoint["tokenizer_cfg"]
        #support for older checkpoints
        with open_dict(model_cfg):
            model_cfg["type"] = "CHECKPOINT"
        
        #load model
        self.load_model(DictConfig(model_cfg))
        
        #quantize layers to load state dict
        quant_cfg = checkpoint.get("quant_cfg", {"quantize": False})
        if quant_cfg["quantize"]:
            #TODO: self.model is set to quantized model before state_dict is loaded
            self.quantize_model(checkpoint["quant_cfg"])
            #skip missing params from checkpoint
            #for some reason, mlp qparams are saved within checkpoint but not the ones from mha
            from brevitas import config
            config.IGNORE_MISSING_KEYS = True
        
        #load state from checkpoint
        load_state_dict_proxy(self.model, checkpoint["model_state_dict"])
        self.optimizer_state_dict = checkpoint["optimizer_state_dict"]
        self.metrics = checkpoint['metrics']


    def save_model(self, cfg: DictConfig):
        #save_checkpoint(cfg, model = cfg.model, dataset = cfg.dataset, optimizer = cfg.optimizer, timestamp, metrics, epoch, model_cfg, tokenizer_cfg)
        #basically the same as save_checkpoint from helpers
        raise NotImplementedError()

    def load_model(self, model_cfg: DictConfig):
        self.model = get_model(model_cfg)
        self.model_cfg = model_cfg

    def quantize_model(self, quant_cfg: DictConfig):
        """
        Quantizes a loaded model with the specified config. It is expected that the quant_cfg is applicable
        to the currently saved 'model' attribute.
        """
        #TODO: discuss if quantizing huggingface models with our framework is possible and a good idea
        #quantize missing layers first
        #TODO: param quant_cfg not really necessary if replace_layers_later is used
        #TODO 2: quantizer used for replace_layers_later could be different from param quant_cfg
        #TODO 3: if any changes to the module have been made, the references to model from quant_cfg need to be updated
        """if self.replace_layers_later is None: 
            log.warning(f'replace_layers_later not tested yet')
            self.quantizer, self.quant_cfg = get_quantizer(quant_cfg, self.model)
            self.model, self.replace_layers_later = self.quantizer.get_quantized_model(self.replace_layers_later)
            self.quant_cfg.model = self.model"""
        if self.replace_layers_later is not None:
            log.warning(f'{self.replace_layers_later=} not supported yet for qat, ignoring option')
        
        self.quantizer, self.quant_cfg = get_quantizer(quant_cfg, self.model)
        self.model, self.replace_layers_later = self.quantizer.get_quantized_model(self.quant_cfg)
        self.quantized = True
        
    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        if labels is not None:
            logits, loss = self.model(idx_cond, labels)
        else:
            logits, loss = self.model(idx_cond)
        return logits, loss

    def compile(self):
        if torch_version >= (2,0):
            self.model = torch_compile(self.model)
        else:
            log.error(f'Cannot compile with torch version: {torch_version} (needs to be >= 2.0)')

model_args_dict = {
    "M": (str,"cls"),
    "s": (int,"args.block_size"),
    "t": (int,"args.vocab_size"),
    "l": (int,"args.n_layer"),
    "h": (int,"args.n_head"),
    "e": (int,"args.n_embd"),
    "A": (str,"args.transformer_active_func"),
    "N": (str,"args.norm_layer"),
    "P": (str,"args.pos_layer"),
}
def update_conf(_config, _str):
    key = _str[0]
    value = _str[1:]
    f, conf_key = model_args_dict.get(key, None)
    if conf_key is not None:
        OmegaConf.update(_config, conf_key, f(value), force_add=True)
    else:
        raise NameError(f"key substring[0] not defined for model args in model.cstr")

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    log.debug(f"get_model config: {model_cfg}")
    if model_cfg.get('cls') is None:
        log.error(f'No model class specified')
        raise KeyError()

    # if model.cstr is present try to infer some model args from this string, then update model.model_name to update placeholder
    cstr = model_cfg.get('cstr')
    if cstr is not None and isinstance(cstr, str) and cstr != "":
        # csr example: Mgpt2-s256-t2048-l2-h4-e512-AReLU-NBatchNormTranspose-Plearned
        for substring in cstr.split("-"):
            update_conf(model_cfg, substring)

    #models need to specify their hyperparameters in init parameter named "config"
    from qtransform import model as _model
    args = model_cfg.get("args")
    model = get_data(log, package_name = _model, class_name = model_cfg.get('cls'), parent_class = nn.Module)
    if args:
        model = model(config = args)
    else:
        #use default args of model
        model = model()
    return model

class PretrainedHuggingfaceQRTModelWrapper(QTRModelWrapper):

    def __init__(self, model_cfg):
        self.load_model(model_cfg)
        self.model_type = ModelType.PRETRAINED

    def from_file(self, path):
        log.warning(f'from_file for pretrained hf models is ignored')

    def save_model(self, cfg: DictConfig):
        #save_checkpoint(cfg, model = cfg.model, dataset = cfg.dataset, optimizer = cfg.optimizer, timestamp, metrics, epoch, model_cfg, tokenizer_cfg)
        raise NotImplementedError()

    def load_model(self, model_cfg: DictConfig):
        self.model = get_hf_pretrained(model_cfg)
        hf_model_cfg = self.model.config
        flash = model_cfg.args.get('flash', False)
        del model_cfg.args #args infered from pretrained model
        activation_function = hf_model_cfg.activation_function
        log.warning(f'Activation function could be wrong ({activation_function})')
        with open_dict(model_cfg):
            model_cfg["args"] = {
                "n_layer" : hf_model_cfg.n_layer,
                "n_head" : hf_model_cfg.n_head,
                "n_embd" : hf_model_cfg.n_embd,
                "dropout" : hf_model_cfg.resid_pdrop, # multiple dropout values possible (resid_pdrop, summary_first_dropout, attn_pdrop, embd_pdrop)
                "bias" : True, # do we use bias inside LayerNorm and Linear layers? # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
                "block_size" : hf_model_cfg.n_ctx,
                "vocab_size" : hf_model_cfg.vocab_size,
                "transformer_active_func": activation_function,
                "norm_layer": "LayerNorm", #should usually be layernorm
                "flash": flash, 
                "single_output": False,
                "use_weight_tying": False, #TODO: find out where you can get that info
                "shift_targets": True
            }
        self.model_cfg = model_cfg

    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Union[Tensor,None]]:
        out = self.model(input_ids = idx_cond, labels = labels) 
        #loss can be None if no labels are supplied
        return (out.logits, out.loss if out.loss is not None else None)

"""
TODO: 
    using onnxruntime-gpu yields CUDA usage of: 
        Self CPU time total: 2.055s
        Self CUDA time total: 7.967ms
    without CUDA usage:
        Self CPU time total: 1.993s
        Self CUDA time total: 8.410ms

    maybe due to: https://onnxruntime.ai/docs/performance/tune-performance/troubleshooting.html#why-is-my-model-running-slower-on-gpu-than-on-cpu
"""
class ONNXQTRModelWrapper(QTRModelWrapper):

    def __init__(self, model_cfg: DictConfig):
        self.load_model(model_cfg)
        self.model_type = ModelType.ONNX
        log.info(f'ONNX model is on: {onnxruntime.get_device()}')

    def load_model(self, model_cfg: DictConfig):
        assert model_cfg.from_file 
        self.from_file(model_cfg.from_file)
        #TODO: context length could be infered from:
        #model.graph.input[0].type.tensor_type.shape.dim[-1]
        self.model_cfg = model_cfg

    def from_file(self, from_file: FromFile):
        path = from_file.get_filepath()
        self.model = load_onnx_model(path)

    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        device = idx_cond.device
        # defaults:
        input_name =  "input"
        run_fn = None
        # determine if finn onnx or qonnx
        _a = re.findall(r"\.finn", str(self.model_cfg.from_file))
        if len(_a) == 1 and _a[0] == ".finn":  # check if model name contains "finn flag" 
            input_name =  "global_in"
            
            # late import for compatibilty iussues on pc2
            from finn.core.onnx_exec import execute_onnx as finn_execute_onnx
            run_fn = finn_execute_onnx
        # elif self.model_cfg.get("runtime") is not None:
        #     if self.model_cfg.get("runtime") == "finn":
        #         input_name =  "global_in"
        #         run_fn = finn_execute_onnx
        #     elif self.model_cfg.get("runtime") == "qonnx":
        #         input_name =  "input"
        #         run_fn = qonnx_execute_onnx
        #     elif self.model_cfg.get("runtime") == "onnx":
        #         input_name =  "input"
        #         run_fn = qonnx_execute_onnx
        else: # defaults
            input_name =  "input"
            from qonnx.core.onnx_exec import execute_onnx as qonnx_execute_onnx
            run_fn = qonnx_execute_onnx

        #log.info(f"input_name for onnx graph run {input_name},  using function {run_fn.__name__}")
        # finn removes batch so we do it here
        if input_name ==  "global_in":
            idx_cond = torch.squeeze(idx_cond)
            #print(idx_cond.cpu().numpy())
            #print(idx_cond.cpu().numpy().size)
            #np.save("global_in.npy", idx_cond.cpu().numpy())

        if labels is not None:
            idict = {input_name: idx_cond.cpu().numpy(), "labels": labels.cpu().numpy()}
            warn_once(log, "labels are givin to external forwards pass wrapper but they are ignored inside onns models atm")
        else:
            idict = {input_name: idx_cond.cpu().numpy()}
        # use infer_shapes()
        #forward pass of gpt model returns the non-softmaxed token predictions
        odict = run_fn(self.model, idict)
        # print(odict)
        if "global_out" in odict.keys():
            #print(odict["global_out"])
            #print(odict["global_out"].size)
            #np.save("global_out.npy", )

            logits = from_numpy(odict["global_out"]).to(device=device)
            logits = torch.unsqueeze(logits, 0)
        else:
            logits = from_numpy(odict["output"]).to(device=device)

        return logits 

    def save_model(self):
        raise NotImplementedError
    
    def to(self, **kwargs):
        pass



"""
    1. new run: specify model config
    2. from checkpoint: model config in checkpoint -> check if hydra model config is specified
    3. from onnx: model config not necessary
    -> cases determined by model.from_file
"""
#either get type from model_cfg or use ModelType Enum as param
def get_model_wrapper(model_cfg: DictConfig) -> QTRModelWrapper:
    assert isinstance(model_cfg.get('type'), str), f'Field "type" within model_cfg not specified'
    model_type = model_cfg.get('type').upper()
    assert hasattr(ModelType, model_type), f'Field "type" ({model_type} not supported)'
    #TODO: when specifying checkpoint, model config not really necessary
    match model_type:
        case "ONNX":
            model = ONNXQTRModelWrapper(model_cfg)
        case "CHECKPOINT":
            model = DynamicCheckpointQTRModelWrapper(model_cfg)
        case "PRETRAINED":
            model = PretrainedHuggingfaceQRTModelWrapper(model_cfg)
        case _:
            log.error(f'Field "type" of model_cfg did not match either ONNX, CHECKPOINT or PRETRAINED.')
            raise KeyError()
    #from_file is loaded within workflow of QRTModelWrapper, no need to explicitly call it here
    #from_file: FromFile = model_cfg.get('from_file')
    #if isinstance(from_file.filename, str) and len(from_file.filename.replace("'", "")) > 0:
    #    model.from_file(from_file)
    return model

