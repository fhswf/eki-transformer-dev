import logging
from typing import Any
log = logging. getLogger(__name__)
from omegaconf import DictConfig
from torch import nn
import torch
import tiktoken
from torch import functional as F

import onnx
import onnxruntime as ort
import numpy as np

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.core.onnx_exec import execute_onnx

def run(cfg : DictConfig):
    """ Inference """
    log.info("========================================")
    log.info("Running Inference on onnx exported model")
    log.info("========================================")
    
    cuda = None
    device = None
    if "cuda" in cfg:
        cuda = cfg.cuda and torch.cuda.is_available()
    else:
        cuda = torch.cuda.is_available()

    torch.manual_seed(cfg.seed)    
    if cuda:
        device = torch.device("cuda")
        cuda_kwargs = {'pin_memory': True,}
        cfg.dataset.dataloader.update(cuda_kwargs)
    else:
        device = torch.device("cpu")
    log.info(f"using device: {str(device)}")
    


    # maybe only do this when it is required, for this howiever is always the case
    from onnx.shape_inference import infer_shapes
    model = ModelWrapper(cfg.run.model_path)    
    infered_shapes = infer_shapes(model.model)
    model = ModelWrapper(infered_shapes)    

    # inspect graph
    #for n in model._model_proto.graph.node:
    #    print(n.name, n.input, n.output)
    #    for i in n.input:
    #        print(model.get_tensor_shape(i), end="")
    #    for o in n.output:
    #        print(model.get_tensor_shape(o), end="")
    #    print("")

    #idict = {"in0" : np.load("in0.npy"), "in1" : np.load("in1.npy")}
    idict = {"input": np.asarray([769, 30, 123])}

    # use infer_shapes()
    odict = execute_onnx(model, idict)
    print(odict)