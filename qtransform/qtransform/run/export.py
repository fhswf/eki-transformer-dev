
from copy import deepcopy
import logging
from typing import Any
import numpy as np
from omegaconf import DictConfig, open_dict
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
    torch.set_printoptions(precision=10)
    device_singleton.device = cfg.device
    device = device_singleton.device

    if device.type == 'cuda':
        cuda_kwargs = {'pin_memory': True,}
        if "dataset" in cfg and cfg.dataset is not None:
            with open_dict(cfg.dataset.dataloader):
                cfg.dataset.dataloader.update(cuda_kwargs)

    from omegaconf import DictConfig, OmegaConf, errors
    try: ## this is so dirty, but for some reason OmegaConf does not work here...
        _run = cfg.run.running_model
    except errors.ConfigAttributeError:
        _run = False
    #export script could have been called directly or from training script
    #the model passed from the training script should be configured completely
    #the model does not exist yet when calling the export script directly
    if  _run:
        log.info("Getting model config from model_wrapper kwargs")
        model_wrapper: DynamicCheckpointQTRModelWrapper = kwargs["model_wrapper"]
    else:
        log.info("Getting model config from hydra config")
        model_wrapper: DynamicCheckpointQTRModelWrapper = get_model_wrapper(cfg.model)
    model: GenericModel = model_wrapper.model
    assert isinstance(model, GenericModel), f'model is not of type GenericModel'
    # fix model? TODO validate this step
    patch_non_affine_norms(model)
    
    # now we need a dataset to calibrate missing quantizers before export 
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
    tokenizer_singleton.tokenizer = cfg.tokenizer
    from qtransform.dataset import DataLoaderWrapper, DatasetSplitType
    dataloader_wrapper = DataLoaderWrapper(cfg.dataset)
    eval_dataloader = dataloader_wrapper.get_loader(DatasetSplitType.EVAL)
     
    model.cpu()
    sample_data = dataloader_wrapper.get_loader(DatasetSplitType.TRAIN)
    for i, data_point in enumerate(sample_data):
        # its a good idea to run some data through the model
        # TODO make calibration run configurable
        if i == 10:
            break
        data_point['input_ids'].cpu()
        model(data_point['input_ids'])
        
    sample_tensor = data_point['input_ids']
    sample_tensor.cpu()
    
    # TODO cleanup this block
    #prepare_and_transform_for_export(cfg, model)
    #by default, save onnx models into current directory
    root_path = cfg.run.get('root_path', os.path.abspath('.'))
    filename = cfg.model.from_file.filename
    if os.path.isabs(filename):
        _, filename = os.path.split(filename)
    model_name = f"{str(cfg.run.export_fn)}_" + filename
    from qtransform.utils.introspection import concat_paths
    model_path = concat_paths([root_path, model_name])
    
    # start export
    log.info("exporting... " + model_name)
    # export params and logs 
    ERROR_LOGS = {
        "qonnx": f'{export_qonnx.__module__}.{export_qonnx.__name__}',
        "qcdq": f'{export_onnx_qcdq.__module__}.{export_onnx_qcdq.__name__}',
        "onnx": f'{export.__module__}.{export.__name__}'
    }
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
    try:
        shape = sample_tensor.clone().detach() #avoid warning from torch, unsure if detaching shape is going to be detrimental
        match cfg.run.export_fn:
            case "qonnx":
                export_qonnx(model, shape, export_path=model_path, **kwargs)
                # export_qonnx(model, shape, export_path="attention.onnx", **kwargs)
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
    # TODO switch onnx execute function when model gets converted by fin before export 
    from qonnx.core.onnx_exec import execute_onnx
    from qonnx.core.modelwrapper import ModelWrapper as QModelWrapper
    onnx_model = QModelWrapper(model_path)
    # run infer shapes on graph just in case?
    from onnx.shape_inference import infer_shapes
    onnx_model = infer_shapes(onnx_model._model_proto)
    # reasign necessary?
    onnx_model = QModelWrapper(onnx_model)
    # run comparson
    # multi_compare_model(model, onnx_model, execute_onnx, sample_data)
   

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


def re_init_quantizers(model, eval_data, num_passes=20):
    """ requires dataset to be loaded. Runs eval set through the model. Without eval mode veing active."""
    @torch.no_grad()
    def _run(model, eval_data, num_passes):
        max_len = len(eval_data)
        run_len = min(max_len, num_passes)
        log.info(f"eval_data len is {max_len}, num_passes set to {num_passes}. Running eval for {run_len}")
        # TODO use cfg.run.eval_iters for eval iter limits when we need it
        for i, vdata in enumerate(eval_data):
            if i > run_len:
                break
            vinputs = None
            vlabels = None
           
            if len(vdata) > 2:
                vinputs = vdata['input_ids']
                vlabels = vdata['labels']
                vattention_mask = vdata['attention_mask']
            elif len(vdata) == 2:
                vinputs, vlabels = vdata
            else:
                log.error(f"unsupported dataloader output. len was {len(vdata)}. ")
                raise NotImplementedError
            #vinputs = vinputs.to(device=device_singleton.device)
            #vinputs = vlabels.to(device=device_singleton.device)
            vinputs.cpu()
            vinputs.cpu()
            voutputs, loss = model(vinputs, vlabels) 
            print(loss)
        return 
    model.train()
    _run(model, eval_data, num_passes)    
    model.eval()
    pass


def patch_non_affine_norms(model: torch.nn.Module):
    """    
    Fixes export issues of normalization layers with disabled affine parameters.
    Somehow the export to ONNX trips when it encounters the weight and bias tensor
    to be 'None'.
    """
    # Check whether a layer is a normalization layer of some supported type
    def is_norm_layer(module):
        # Set of normalization layer (bases) which maybe need to be patched
        norm_layers = {
            # All BatchNorm and InstanceNorm variants derive from this baseclass
            torch.nn.modules.batchnorm._NormBase,  # noqa: Access to _NormBase
            # LayerNorm has a unique implementation
            torch.nn.LayerNorm,
        }
        # Check the module against all supported norm layer types
        return any(isinstance(module, norm) for norm in norm_layers)

    # Iterate all modules in the model container
    for name, module in model.named_modules():
        # If the module is a normalization layer it might require patching the
        # affine parameters
        if is_norm_layer(module):
            # Check whether affine scale parameters are missing
            if hasattr(module, "weight") and module.weight is None:
                # There need to be running statistics to patch the scales
                if hasattr(module, "running_var"):
                    # Patch the affine bias by all 1 tensor of the same shape,
                    # type and device as the running variance
                    module.weight = torch.nn.Parameter(
                        torch.ones_like(module.running_var)
                    )
            # Check whether affine bias parameters are missing
            if hasattr(module, "bias") and module.bias is None:
                # There need to be running statistics to patch the scales
                if hasattr(module, "running_mean"):
                    # Patch the affine bias by all 0 tensor of the same shape,
                    # type and device as the running mean
                    module.bias = torch.nn.Parameter(
                        torch.zeros_like(module.running_var)
                    )
    # Return the patched model container
    return model


def compare_model_outputs(torch_model, onnx_model, onnx_run_function, input_tensor):
    """
    Compares the output of model with input_tensor as a sample. Generates diff tensor.
    """ 
    # model.to(device=device) always run validation on cpu?
    torch_model.cpu()
    input_tensor.cpu()
    torch_model.eval()
    
    o1 = model(sample_tensor)
    
    # Save the input and output data for verification purposes later
    in_tensor  = sample_tensor.cpu()
    out_tensor = o1[0].cpu()
    np.save(model_name + ".inp.npy", in_tensor.detach().numpy())
    np.save(model_name + ".out.npy", out_tensor.detach().numpy())
    
    o2 =  onnx_run_function(onnx_model, {"input": input_tensor.cpu().numpy()})
    o2 = torch.from_numpy(o2["output"])
    o1 = o1[0]
    o1 = o1.detach().cpu()
    o2 = o2.detach().cpu()
    
    torch.set_printoptions(precision=10)
    print(torch.squeeze(o1,0))
    print(torch.squeeze(o2,0))
    
    if not torch.equal(o1, o2):
        log.warning(f"sample tensor after export does not equal model output before export")
    if not torch.allclose(o1, o2,atol=1e-3):
        log.error(f"sample tensor after export is not close to model output before export")
        import matplotlib.pyplot as plt
        #plt.show(torch.squeeze(0, sample_tensor).detach().numpy(),interpolation='none')
        #plt.savefig('input.png')
        plt.imshow(torch.squeeze(o1,0), interpolation='none')
        plt.savefig('model.png')
        plt.imshow(torch.squeeze(o2,0), interpolation='none')
        plt.savefig('onnx.png')
        mask = torch.isclose(o1, o2,atol=1e-3)
        
        plt.imshow(torch.squeeze(mask,0), interpolation='none')
        plt.savefig('mask.png')
        oof = (mask == False).nonzero(as_tuple=False)
        print(oof)
        print(sample_tensor[0, oof[1]])
        print(o1[0, oof[1], oof[2]])
        print(o2[0, oof[1], oof[2]])
       
    pass

def multi_compare_model(torch_model_wrapper: GenericModel, onnx_model, onnx_run_function, sample_data):
    """
    Runs multiple comparisons of torch_model and onnx_model to validate the correct model output.
    compares multiple random input tensors and muliple data_points from the evaluation data_loader.
    Logs results to wandb.
    """
    assert isinstance(torch_model_wrapper, GenericModel), f'model is not of type GenericModel (qtransform specific)'
    
    raise NotImplementedError
    
    for i, data_point in enumerate(_data):
        jjj = data_point
        break
    print("==========")
    print(jjj["input_ids"][0][None,:])
    sample_tensor = jjj["input_ids"][0][None,:]
    
    input_dim = (1, model.config.block_size)
    #input_dim = (1, checkpoint['model_cfg']['args']['block_size'])
    max_token_id = model.config.vocab_size
    #max_token_id = checkpoint['model_cfg']['args']['vocab_size']
    random_sample_tensor = torch.randint(0, max_token_id, input_dim, dtype=int).to(device=device)
    pass