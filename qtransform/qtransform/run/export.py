
from copy import deepcopy
import logging
from typing import Any
import numpy as np
from omegaconf import DictConfig
import hydra
import os
from qtransform import device_singleton
from qtransform.utils.helper import load_checkpoint
from qtransform.model import ModelArgs, GenericModel, get_model_wrapper, DynamicCheckpointQTRModelWrapper
import torch
from torch import nn
from brevitas.export import export_onnx_qcdq, export_qonnx, export_brevitas_onnx
from brevitas import nn as qnn
from torch.onnx import export
from datetime import datetime

log = logging.getLogger(__name__)

def run(cfg: DictConfig, **kwargs):
    """ exports a trained model to QONNX or others?"""
    log.info("================")
    log.info("Exporting Model")
    log.info("================")
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    log.info(f"time is: {timestamp}")
    log.debug(f'Run config: {cfg.run}')
    model = None

    device_singleton.device = cfg.device
    device = device_singleton.device

    from omegaconf import DictConfig, OmegaConf, errors
    try: ## this is so dirty, but for some reason OmegaConf does not work here...
        _run = cfg.run.running_model
    except errors.ConfigAttributeError:
        _run = False
    #export script could have been called directly or from training script
    #the model passed from the training script should be configured completely
    #the model does not exist yet when calling the export script directly
    if  _run:
        model_wrapper: DynamicCheckpointQTRModelWrapper = kwargs["model_wrapper"]
    else:
        model_wrapper: DynamicCheckpointQTRModelWrapper = get_model_wrapper(cfg.model)
    model: GenericModel = model_wrapper.model
    assert isinstance(model, GenericModel), f'model is not of type GenericModel'
    #log.debug(f"Model structure: {model}")
    #log.debug(f"Model config from checkpoint: {checkpoint['model_cfg']}")
    input_dim = (1, model.config.block_size)
    #input_dim = (1, checkpoint['model_cfg']['args']['block_size'])
    max_token_id = model.config.vocab_size
    #max_token_id = checkpoint['model_cfg']['args']['vocab_size']
    sample_tensor = torch.randint(0, max_token_id, input_dim, dtype=int).to(device=device)

    """
    export function maps to torch onnx export:
    # Export the model
        torch.onnx.export(torch_model,  # model being run
        x,                              # model input (or a tuple for multiple inputs)
        "model.onnx",        # where to save the model (can be a file or file-like object)
        export_params=True,             # store the trained parameter weights inside the model file
        opset_version=10,               # the ONNX version to export the model to
        do_constant_folding=True,       # whether to execute constant folding for optimization
        input_names = ['input'],        # the model's input names
        output_names = ['output'],      # the model's output names
        dynamic_axes={'input' : {0 : 'batch_size'},    # variable length axes
                    'output' : {0 : 'batch_size'}})
    """
    kwargs = {
        "input_names" :['input', 'offsets'],   # the model's input names
        "output_names" : ['output'],         # the model's output names
        #"dynamic_axes" :{'input' : {0 : 'batch_size'},    # variable length axes
        #                'output' : {0 : 'batch_size'}},
        "opset_version": cfg.run.opset_version,  
        "export_params": True,  
        "do_constant_folding": True
    }
    #prepare_and_transform_for_export(cfg, model)
    #by default, save onnx models into current directory
    root_path = cfg.run.get('root_path', os.path.abspath('.'))
    #TODO: makedirs in ~/.qtransform directory deletes datasets 
    #if not os.path.exists(root_path):
    #    log.debug(f'Creating directory: "{root_path}')
    #    os.makedirs(root_path.replace('~', os.path.expanduser('~')), exist_ok = True)
    #elif not os.path.isdir(root_path):
    #    log.error(f'root_path {root_path} is not a directory.')
    #    raise ValueError()
    filename = cfg.model.from_file.filename
    if os.path.isabs(filename):
        _, filename = os.path.split(filename)
    model_name = f"{str(cfg.run.export_fn)}_" + filename
    from qtransform.utils.introspection import concat_paths
    model_path = concat_paths([root_path, model_name])

    log.info("exporting... " + model_name)
    ERROR_LOGS = {
        "qonnx": f'{export_qonnx.__module__}.{export_qonnx.__name__}',
        "qcdq": f'{export_onnx_qcdq.__module__}.{export_onnx_qcdq.__name__}',
        "onnx": f'{export.__module__}.{export.__name__}'
    }

    model.eval()
    o = model(sample_tensor)
    print(o)
    # Save the input and output data for verification purposes later
    in_tensor  = sample_tensor.cpu()
    out_tensor = o[0].cpu()
    np.save(model_name + ".inp.npy", in_tensor.detach().numpy())
    np.save(model_name + ".out.npy", out_tensor.detach().numpy())

    # TODO 
    # verfiy tensor output before and after qonnx conversion

    #only write something into pipe if no errors occur
    try:
        shape = sample_tensor.clone().detach() #avoid warning from torch, unsure if detaching shape is going to be detrimental
        match cfg.run.export_fn:
            case "qonnx":
                export_qonnx(model, shape, export_path=model_path, **kwargs)
                export_qonnx(model, shape, export_path="attention.onnx", **kwargs)
            case "qcdq":
                export_onnx_qcdq(model, shape, export_path=model_path, **kwargs)
            case "onnx":
                export(model, shape, model_path, **kwargs)
            case _:
                log.error(f'Supported export functions: {ERROR_LOGS.keys()}')
                raise ValueError()
    except Exception:
        log.error(f"Export via {ERROR_LOGS[cfg.run.export_fn]} failed, reason", exc_info=True)
        raise RuntimeError()
    
    # these step sare to verify model outpus matches onnx export
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.core.modelwrapper import ModelWrapper as QModelWrapper
    onnx_model = QModelWrapper(model_path)
    from onnx.shape_inference import infer_shapes
    onnx_model = infer_shapes(onnx_model._model_proto)
    onnx_model = QModelWrapper(onnx_model)
    o =  execute_onnx(onnx_model, {"input": sample_tensor.cpu().numpy()})
    print(o)

def search_for_weight(model, module: nn.Module)->(bool, str):
    paramname = None
    has_standart_weight = False
    for n,m in model.named_parameters():
        if n.endswith(".weight"):
            has_standart_weight = True
            paramname = n
    return has_standart_weight, paramname
        
from qtransform.quantization.quant_bn import replace_bn, CustomBatchNorm1d, QuantBatchnorm1d
def auto_merge_layers(cfg, model: torch.nn.Module, inplace=False, qat=True):
    """
    Should be used wiht caution. Auto merging layers only works if all layers are sequential. 
    Which is to say:  batchnorm or layernorm appear directly after some linear tranformation.
    """
    model: torch.nn.Module = model if inplace else deepcopy(model)
    #last_module: torch.nn.Module = None
    #param_name = None
    for mn, module in model.named_modules():

        # merge if applicable
        # currently, only batchnorm is merged
        if isinstance(module, nn.modules.batchnorm._NormBase):
            log.debug("=========")    
            module = replace_bn(module, qat=qat)
            #raise NotImplementedError(f'merge_bn is currently being refactored')
            """if isinstance(module, qnn.QuantMultiheadAttention):
                merge_bn_mha(last_module, model)
                # TODO remove bn layer connect (and connect nodes)?
            elif isinstance(last_module, nn.MultiheadAttention):
                raise NotImplementedError
            elif param_name is not None: ## means we found weight by name
                log.info(f"Last Layer with weigts was {last_module}, trying to merge {module} weights")
                qnn.utils.merge_bn(last_module, model)
            else:
                log.error(f"cant merge norm layer because we dont know what to merge it into. Module is {module}")

        # log last layer with weights
                
        # log.debug(module)
        #n will be somth like this:  transformer.layer.0.attn.c_proj.weight 
        if isinstance(module, nn.MultiheadAttention) or isinstance(module, qnn.QuantMultiheadAttention):
            log.debug(f"last layer with nn.MultiheadAttention or qnn.QuantMultiheadAttention {mn}")
            last_module = module
            param_name = None
        else:
            yes, param_name = search_for_weight(model, module)
            if yes:
                log.debug(f"last layer with weights {mn} {param_name}")
                last_module = module
                #log.debug(last_module)"""

    #raise NotImplementedError
    return model
def prepare_and_transform_for_export(cfg, model: torch.nn.Module, inplace=False, qat=True):
    """
    used to merge Layers like BatchNorm. Layers that are not quantized if they shall be quantized could maybe create a warning of some sorts? 
    """
    if True: #cfg.run.auto_merge:
        return auto_merge_layers(cfg, model, inplace, qat)
    else:
        # TODO use some merge config. where for evry norm layer a merge layer is specified.
        raise NotImplementedError

    

    