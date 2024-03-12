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

log = logging.getLogger(__name__)


class ModelType(Enum):
    ONNX = 0
    CHECKPOINT = 1

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
    pretrained = model_cfg.get('pretrained', False)
    if pretrained: 
        assert model_cfg.version is not None, f'pretrained model version needs to be specified'
        model = GPT2LMHeadModel.from_pretrained(model_cfg.version)
        #necessary for dataloader to retrieve exactly one full context input
        with open_dict(model_cfg):
            model_cfg.args.block_size = model.config.n_positions
            model_cfg.args.vocab_size = model.config.vocab_size
            model_cfg.args.calc_loss_in_model = True
        return model
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

def get_onnx(path: str) -> ModelWrapper:
    """
    Alias for load_onnx_model from qtransform.utils.helper.
    """
    return load_onnx_model(path)

class QTRModelWrapper(ABC):
    """QTRModelWrapper instead of ModelWrapper to avoid naming conflicts with ONNX models"""
    #TODO: properties
    model: Union[ModelWrapper, nn.Module] 
    model_type: str

    def __init__(self, model_cfg: DictConfig):
        self.model_cfg = model_cfg
        self.load_model(model_cfg)
        #self.config = None

    @abstractmethod
    def load_model(self, model_cfg: DictConfig):
        pass

    @abstractmethod
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

#TODO: pretrained huggingface models
class DynamicCheckpointQTRModelWrapper(QTRModelWrapper):

    def __init__(self, model_cfg):
        super().__init__(model_cfg=model_cfg)
        self.model_type = ModelType.CHECKPOINT

    def from_file(self, path):
        cfg = DictConfig({
            "run.from_checkpoint": path
        })
        from_epoch, checkpoint = load_checkpoint(cfg)
        model_cfg = checkpoint["model_cfg"]
        model = get_model(DictConfig(model_cfg))
        self.model = model
        self.model_cfg = model_cfg


    def save_model(self, cfg: DictConfig):
        #save_checkpoint(cfg, model = cfg.model, dataset = cfg.dataset, optimizer = cfg.optimizer, timestamp, metrics, epoch, model_cfg, tokenizer_cfg)
        raise NotImplementedError()

    #TODO: no idea if modelwrapper should be able to point to different models
    def load_model(self, model_cfg: DictConfig):
        self.model = get_model(model_cfg)
        self.model_cfg = model_cfg

    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        if labels is not None:
            logits, loss = self.model(idx_cond, labels)
        else:
            logits, loss = self.model(idx_cond)
        return logits, loss
    


class ONNXQTRModelWrapper(QTRModelWrapper):

    def __init__(self, model_cfg: DictConfig):
        super().__init__(model_cfg=model_cfg)
        self.model_type = ModelType.ONNX

    def load_model(self, model_cfg: DictConfig):
        self.from_file(model_cfg.from_file)
        self.model_cfg = model_cfg

    #TODO: model_cfg points to different model still, from_file does not need model_cfg though
    def from_file(self, path: str): 
        self.model = get_onnx(path)

    def forward(self, idx_cond: Tensor, labels = None) -> Tuple[Tensor, Tensor]:
        device = idx_cond.device
        idict = {"input": idx_cond.cpu().numpy(), "labels": labels.cpu().numpy()}
        # use infer_shapes()
        #forward pass of gpt model returns the non-softmaxed token predictions
        if labels is not None:
            log.warning("labels are givin to external forwards pass wrapper but they are ignored atm for onnx runs")
        odict = execute_onnx(self.model, idict)
        log.warning(odict)
        logits = from_numpy(odict["output"]).to(device=device)
        return logits 

    def save_model(self):
        raise NotImplementedError


def get_model_wrapper(model_cfg: DictConfig) -> QTRModelWrapper:
    model_type = model_cfg.get('type').upper()
    match model_type:
        case "ONNX":
            return ONNXQTRModelWrapper(model_cfg)
        case "CHECKPOINT":
            return DynamicCheckpointQTRModelWrapper(model_cfg)
        case _:
            log.error(f'Field "type" of model_cfg did not match either ONNX or CHECKPOINT')
            raise KeyError()

