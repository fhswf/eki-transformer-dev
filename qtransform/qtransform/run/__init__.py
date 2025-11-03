from typing import Tuple, Union, List, Dict, Any
from omegaconf import DictConfig, open_dict
import os
from dataclasses import dataclass
from qtransform.tokenizer.tokenizer import Tokenizer
from qtransform.utils.helper import load_checkpoint, load_onnx_model
from qtransform.utils.introspection import get_classes
from qtransform.model import QTRModelWrapper, ModelType
import torch
from torch import nn
import torch.nn.functional as F
from qonnx.core.onnx_exec import execute_onnx
from qonnx.core.modelwrapper import ModelWrapper
from enum import Enum
from logging import getLogger

log = getLogger(__name__)