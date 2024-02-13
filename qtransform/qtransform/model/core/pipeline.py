from typing import Callable
from transformers import Pipeline
from qtransform.dataset.tokenizer import Tokenizer
import torch
import logging
log = logging.getLogger(__name__)

def preprocess_pipe(tokenizer: Tokenizer, device, padding = None) -> Callable:
    """
    Returns a function for text input, that returns a tensor. 
    If input_dim_pad is not None and a number, input will be padded
    """
    def _preprocess(text):
        input_ids = tokenizer.encode(text)
        input_ids = (torch.tensor(input_ids, dtype=torch.long, device=device)[None, ...])
        if padding is not None:
            pad_to = None
            if isinstance(padding, int):
                pad_to = padding
            if isinstance(padding, tuple):
                # assume second dim is text/block length aka batch first
                pad_to = padding[1]
            if pad_to is None:
                log.error(f"pad of {padding} is not valid")
                raise RuntimeError            
            pad_id = tokenizer.get_pad_id()
            pad_length = pad_to - input_ids.size()[1]
            input_ids = torch.nn.functional.pad(input_ids, (pad_length, 0), mode='constant', value=pad_id)
        return input_ids
    return _preprocess


class QTransformGPTPipeline(Pipeline):
    """
    
    HUgginface Pipeline API. Note that this will break for qonnx and also does not work atm.
    Problems are: Our own tokenizer such as tiktoken does not share the same api.
    TODO register this locally so we can use it in a notebook o smth.  
    """
    def _sanitize_parameters(self, **kwargs):
        preprocess_kwargs = {}
        if "maybe_arg" in kwargs:
            preprocess_kwargs["maybe_arg"] = kwargs["maybe_arg"]
        return preprocess_kwargs, {}, {}

    def preprocess(self, inputs, maybe_arg=2):
        model_input = Tensor(inputs["input_ids"])
        return {"model_input": model_input}

    def _forward(self, model_inputs):
        # model_inputs == {"model_input": model_input}
        outputs = self.model(**model_inputs)
        # Maybe {"logits": Tensor(...)}
        return outputs

    def postprocess(self, model_outputs):
        best_class = model_outputs["logits"].softmax(-1)
        return best_class