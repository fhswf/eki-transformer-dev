from typing import Callable
import torch
#modules which perform basic torch operations in order to avoid symbolic tracing error
#TODO: generic class creator of form:
"""

    class TorchOperator(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._is_leaf_module = True

        def forward(self, x):
            return torch.<torch_operator>(x)


"""

def create_leaf_module(func: Callable) -> torch.nn.Module:
    """
    Function that creates a torch Module from a torch operator (such as neg, sum, pow etc.) in order to avoid Proxy
    errors when trying to symbolically trace a module for PTQ. Every model that should be used for PTQ needs to have this function instead of
    calling the torch operators directly.

    Arguments:
        func: torch function
    Returns:
        torch.nn.Module wrapping the function
    """
    assert getattr(torch, func.__name__, None) is not None, f'Function {func.__name__} is not a torch function.'
    leaf_module: torch.nn.Module = type(
                        "Torch" + func.__name__ + "LeafModule",
                        (torch.nn.Module, ), 
                        {"_is_leaf_module": True}
                    )()
    def forward(x):
        return func(x)
    #add forward function
    setattr(leaf_module, "forward", func)
    return leaf_module