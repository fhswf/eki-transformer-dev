from omegaconf import DictConfig, open_dict
from qtransform.classloader import get_data
from torch import nn, Tensor, from_numpy
import logging
from dataclasses import dataclass
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
# maybe only do this when it is required, for this howiever is always the case
from onnx.shape_inference import infer_shapes
from enum import Enum
from qtransform.utils.helper import load_checkpoint, save_checkpoint, load_onnx_model
from typing import Union, Tuple
from abc import ABC, abstractmethod
import transformers
import onnxruntime

log = logging.getLogger(__name__)

#TODO: dataclass for model args (block_size, embd_dim, vocab_size ...)
#TODO 2: QRTModelWrapper for huggingface models


class ModelType(Enum):
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

def get_model(model_cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """
    log.debug(f"get_model config: {model_cfg}")
    if model_cfg.get('cls') is None:
        log.error(f'No model class specified')
        raise KeyError()
    from qtransform import model as _model
    args = model_cfg.get("args")

    #models need to specify their hyperparameters in init parameter named "config"
    model = get_data(log, package_name = _model, class_name = model_cfg.get('cls'), parent_class = nn.Module)
    #construct model if no args have been given
    if args:
        model = model(config = args)
    else:
        model = model()
    return model

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

def get_onnx(path: str) -> ModelWrapper:
    """
    Alias for load_onnx_model from qtransform.utils.helper.
    """
    return load_onnx_model(path)

#basically the same as GPTConfig, TODO: use GPTConfig structure for other models
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

@dataclass
class ModelConfig():
    type: str #ONNX or DYNAMIC_CHECKPOINT
    cls: str
    calc_loss_in_model: bool
    args: ModelArgs
    from_file: str = None #torch checkpoint or ONNX model path

    def __setattr__(self, name, value):
        if name == "args" and not isinstance(value, ModelArgs):
            value = ModelArgs(**value)
        self.__dict__[name] = value

class QTRModelWrapper(ABC):
    """QTRModelWrapper instead of ModelWrapper to avoid naming conflicts with ONNX models"""
    #TODO: properties
    model: Union[ModelWrapper, nn.Module] 
    model_type: ModelType
    _model_cfg: ModelConfig #missing args from config are replaced with default values of dataclass

    def __init__(self, model_cfg: Union[ModelConfig, DictConfig]):
        self.load_model(model_cfg)

    @property
    def model_cfg(self):
        return self._model_cfg
    
    @model_cfg.setter
    def model_cfg(self, value: Union[ModelConfig, DictConfig]):
        self._model_cfg = value if isinstance(value, ModelConfig) else ModelConfig(**value)

    @abstractmethod
    def load_model(self, model_cfg: Union[ModelConfig, DictConfig]):
        pass

    @abstractmethod
    #TODO: inference needs model cfg even though it only needs pathname to model
    #      ONNX models need model property to get config though
    def from_file(self, path: str):
        pass

    def __call__(self, idx_cond: Tensor, labels = None):
        return self.forward(idx_cond, labels)
    
    @abstractmethod
    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        pass

    @abstractmethod
    def save_model(self):
        raise NotImplementedError
    
    #mainly to avoid conflicts with onnx models
    def to(self, **kwargs):
        self.model.to(**kwargs)

class DynamicCheckpointQTRModelWrapper(QTRModelWrapper):

    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)
        self.model_type = ModelType.CHECKPOINT
    
    def from_file(self, path):
        cfg = DictConfig({
            "run": {
                "from_checkpoint": path
            }
        })
        from_epoch, checkpoint = load_checkpoint(cfg)
        model_cfg = checkpoint["model_cfg"]
        #support for older checkpoints
        with open_dict(model_cfg):
            model_cfg["type"] = "CHECKPOINT"
        model = get_model(DictConfig(model_cfg))
        self.model = model
        self.model_cfg = model_cfg


    def save_model(self, cfg: DictConfig):
        #save_checkpoint(cfg, model = cfg.model, dataset = cfg.dataset, optimizer = cfg.optimizer, timestamp, metrics, epoch, model_cfg, tokenizer_cfg)
        raise NotImplementedError()

    def load_model(self, model_cfg: DictConfig):
        self.model = get_model(model_cfg)
        self.model_cfg = model_cfg

    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        if labels is not None:
            logits, loss = self.model(idx_cond, labels)
        else:
            logits, loss = self.model(idx_cond)
        return logits, loss
    
class PretrainedHuggingfaceQRTModelWrapper(QTRModelWrapper):

    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)
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
        super().__init__(model_cfg=model_cfg)
        self.model_type = ModelType.ONNX
        self.ONNX_LABELS_WARNING = True
        log.info(f'ONNX model is on: {onnxruntime.get_device()}')

    def load_model(self, model_cfg: DictConfig):
        self.from_file(model_cfg.from_file)
        #TODO: context length could be infered from:
        #model.graph.input[0].type.tensor_type.shape.dim[-1]
        self.model_cfg = model_cfg

    #TODO: model_cfg points to different model still, from_file does not need model_cfg though
    def from_file(self, path: str): 
        self.model = get_onnx(path)

    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        device = idx_cond.device
        idict = {"input": idx_cond.cpu().numpy(), "labels": labels.cpu().numpy()}
        # use infer_shapes()
        #forward pass of gpt model returns the non-softmaxed token predictions
        if labels is not None and self.ONNX_LABELS_WARNING:
            log.warning("labels are given to external forwards pass wrapper but they are ignored atm for onnx runs. Suppressing this warning")
            self.ONNX_LABELS_WARNING = False
        odict = execute_onnx(self.model, idict)
        logits = from_numpy(odict["output"]).to(device=device)
        return logits 

    def save_model(self):
        raise NotImplementedError
    
    def to(self, **kwargs):
        pass

#either get type from model_cfg or use ModelType Enum as param
#TODO 2: redundancies in model and run fields (from_file, checkpoint_dir)
def get_model_wrapper(model_cfg: DictConfig) -> QTRModelWrapper:
    model_type = model_cfg.get('type').upper()
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
    from_file = model_cfg.get('from_file')
    if isinstance(from_file, str) and len(from_file.replace("'", "")) > 0:
        model.from_file(from_file)
    return model

