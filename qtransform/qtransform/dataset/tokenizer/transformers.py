from typing import List
from qtransform.dataset.tokenizer import Tokenizer
from numpy import memmap
from omegaconf import DictConfig, open_dict
import logging
import transformers
from datasets import Dataset#, PreTrainedTokenizer, PreTrainedTokenizerFast
from qtransform.classloader import get_data
from re import match, search, compile
from qtransform.utils.introspection import get_classes
import sys

log = logging.getLogger(__name__)

class TransformersTokenizer(Tokenizer):
    """
        Uses transformers as the tokenizer of choice.
        You can choose between the "full" version or the fast version. Differences can be read on 
        https://huggingface.co/docs/transformers/main_classes/tokenizer#tokenizer (from 13.12.2023)
    """
    def __init__(self, tokenizer_cfg, memmap: memmap = None):
        super().__init__(tokenizer_cfg, memmap)
        if self.tokenizer_cfg.get("fast") is None:
            log.warning(f'Missing key "fast" in transformers tokenizer config. Defaulting to False.')
            with open_dict(self.tokenizer_cfg):
                self.tokenizer_cfg.fast = False
        #get pretrainedtokenizer classes and their model encodings
        parent_class = transformers.PreTrainedTokenizerFast if self.tokenizer_cfg.fast else transformers.PreTrainedTokenizer
        pretrained_tokenizers = get_classes(transformers.models, parent_class)
        self.tokenizer = None
        for tokenizer_cls_name, tokenizer_cls in pretrained_tokenizers.items():
            encodings = tokenizer_cls.max_model_input_sizes 
            if self.tokenizer_cfg.encoding in encodings:
                self.tokenizer = tokenizer_cls.from_pretrained(self.tokenizer_cfg.encoding)
        log.debug(f'Using tokenizer class: {self.tokenizer.__class__.__name__} with encoding: {self.tokenizer_cfg.encoding}')
        if self.tokenizer is None:
            log.error(f'No transformers tokenizer found for encoding: {self.tokenizer_cfg.encoding} and fast={self.tokenizer_cfg.fast}')
            raise KeyError()
    
    def tokenize_memmap(self, text: str):
        super().tokenize_memmap(text) #arg checking
        offset = self.num_tokens
        tokens = self.tokenize(text)
        self.memmap[offset : offset + len(tokens)] = tokens

    def tokenize(self, text) -> List[int]:
        #truncation is not performed in this case as only the ids are important currently
        tokens: list[int] = self.tokenizer(text)["input_ids"]
        self.num_tokens += len(tokens)
        return tokens

    def decode(self, idx: List[int]) -> str:
        return self.tokenizer.decode(idx)

    def save_metadata(self, filepath: str):
        # save the meta information as well, to help us encode/decode later
        meta = {
            'max_token_value': self.max_token_value,
            'encoding': self.tokenizer_cfg.encoding,
            'dtype': self.tokenizer_cfg.dtype,
            'fast': self.tokenizer_cfg.fast
        }
        self._save_metadata(filepath, meta)