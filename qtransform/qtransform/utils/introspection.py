from types import ModuleType
import logging
import pkgutil, inspect, importlib
import sys
from os.path import join
from numpy import dtype as np_dtype
from os.path import expanduser

log = logging.getLogger(__name__)
def get_classes(module: ModuleType, parent_class: type):
    """
    'module' should be either a list of paths to look for
    modules in or a python module.
    """
    # type save guard, maybe relevant later...
    paths = None 
    package = None
    if isinstance(module, ModuleType):
        paths = module.__path__
        package = module.__name__
    else:
        log.error(f"module {module} is not an importable ModuleType")
        raise TypeError
    
    classes = {}
    for p in pkgutil.iter_modules(paths):
        m = importlib.import_module(package + "." + p[1]) #p[1] is the module name
        for name, obj in inspect.getmembers(m):
            if parent_class is not None: 
                if inspect.isclass(obj) and issubclass(obj, parent_class):
                    if name not in classes:
                        if hasattr(log, "trace"): log.trace(f"{parent_class.__name__} class found: {name}={obj}")
                        classes[name] = obj
            elif inspect.isclass(obj):
                if name not in classes:
                    if hasattr(log, "trace"): log.trace(f"{parent_class.__name__} class found: {name}={obj}")
                    classes[name] = obj
    return classes
    
def _get_module(module_name, package_name = None, scope = None):
    """
    find and import the module "module_name" dynamically either in current directory/path if scope variable is set to the path
    or from all known python modules (sys.path) if scope = None
    in order to correctly import a module with importlib.import_module, the name of the package in python notation must be provided
    (package.).
    package_name  is the name of the package in python notation (__name__), 
    scope is the path to the package (__path__)
    """
    if scope == None:
        scope = sys.path
    log.debug(f'searching for module {module_name} within scope: {scope}')
    #__path__ makes the search limit to only the current module, not all modules under sys.path
    l = list(filter(lambda x: x.name==module_name, pkgutil.iter_modules(scope)))
    log.debug(f"Found dataset module: {l}")
    if len(l) == 0:
        log.error(f"Dataset type module_name: {module_name} not found")
        log.error(f"Options are: {list(map(lambda x:x.name, pkgutil.iter_modules(scope)))}")
        raise KeyError
    if len(l) > 1:
        log.critical(f"Found more than one module to import for {module_name}, is something wrong with the search path?")
        raise KeyError
    _module_name = l[0].name
    if package_name == None:
        #import the module found from the first package in sys.path
        return importlib.import_module(_module_name)
    return importlib.import_module('..' + _module_name, package_name+'.')

def concat_paths(paths: list) -> str:
    """
        Concatinates paths of a list by joining them with os.path.join.
        In order to avoid issues, the shortcut ~ is translated into the home directory of the user 
        executing the module.
    """
    if paths == None or len(paths) == 0:
        raise ValueError("Error, cannot concatinate paths when argument is empty.")
    main_path = ""
    paths[0] = paths[0].replace('~', expanduser('~'))
    for path in paths:
        main_path = join(main_path, path)
    return main_path

def get_dtype(dtype_alias: str) -> np_dtype:
    dtype = None
    try:
        dtype = np_dtype(dtype_alias)
    except:
        log.critical(f'Datatype {dtype_alias} not found within numpy datatype scope')
        raise KeyError()
    return dtype

from typing import Union, get_origin, get_args, List, Dict
def get_optional_type(_type):
    """
        Unwraps a type which might be encapsulated in type "Optional" from the typing package.
    """
    field_origin = get_origin(_type)
    if field_origin is Union:
        field_type = get_args(_type)
        #get first datatype of union, assuming that is is Optional, the second type is None anyway
        origin_type = field_type[0] if len(field_type) == 2 else field_type#and isinstance(field.type, Union) else field_type
    elif field_origin is None:
        #type was never wrapped
        origin_type = _type
    else: 
        origin_type =  field_origin
    return origin_type

def concat_strings(strings: List[str]) -> str:
    """
        Concats a list of immutable strings by joining them together. 
    """
    return ''.join(strings)

from torch.nn import Module
def get_devices(model: Module) -> Dict[str, List[int]]:
    """
        Returns a dictionary containing the layers of a model along with the devices of each tensor within each layer.
        The devices are represented as an integer, -1 means that the tensor is located on the cpu. Tensor.get_device() is used
        for that. The name of a layer can be nested (transformer.layer.1.mha).
        Example: 
        {
            "linear": [-1, -1]
        } 
        for a simple model containing one layer with two tensors located on the cpu.

        Keep in mind that only the devices of instances of nn.Module and nn.parameter.Parameter are returned, regular Tensors are not.
        Example: 

            class foo(nn.Module):
                def __init__(self, dim: tuple):
                    super().__init__()
                    self.attn_mask = torch.tril(torch.ones(dim))
                def forward(self, input):
                    pass
        
        The parameter dictionary of an instance of class foo is empty, as attn_mask is a Tensor which does not appear in parameters().
        A simple fix for that is to make the attn_mask attribute of type torch.nn.parameter.Parameter with by doing:
            self.attn_mask = torch.nn.parameter.Parameter(torch.tril(torch.ones(dim)))
        Doing that will make device of attn_mask appear within the dictionary.
    """
    devices = dict()
    #go through all parameters of model
    
    #go through all sublayers
    for module_name, module in model.named_modules():
        device_module = list()
        for parameter in module.parameters():
            device_module.append(parameter.get_device())
        devices[module_name] = device_module
    
    return devices