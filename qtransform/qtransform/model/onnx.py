import math
import inspect
from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import functional as F
from qtransform.model.modules import LayerNorm, TransformerBlock
from qtransform.model import modules as custom_nn
from qtransform.model import ModelInfoMixin
from brevitas import nn as qnn
import logging
log = logging.getLogger(__name__)

from abc import ABC, abstractmethod

class ONNX_Model(ModelInfoMixin):
    def __init__(self):
        super().__init__()
        pass

