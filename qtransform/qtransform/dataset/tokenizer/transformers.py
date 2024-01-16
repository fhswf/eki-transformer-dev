from typing import List, Dict, Any
from qtransform.dataset.tokenizer import Tokenizer, Metadata
from numpy import memmap, array
from omegaconf import DictConfig, open_dict
import logging
import transformers
from datasets import Dataset#, PreTrainedTokenizer, PreTrainedTokenizerFast
from qtransform.classloader import get_data
from re import match, search, compile
from qtransform.utils.introspection import get_classes
import sys
from dataclasses import dataclass, asdict

FAST = True #by default, use fast implementation of tokenizers

@dataclass
class TransformersMetadata(Metadata):
    module: str = "transformers"
    fast: bool = FAST

log = logging.getLogger(__name__)

class TransformersTokenizer(Tokenizer):
    """
        Uses transformers as the tokenizer of choice.
        You can choose between the "full" version or the fast version. Differences can be read on 
        https://huggingface.co/docs/transformers/main_classes/tokenizer#tokenizer (from 13.12.2023)
    """
    def __init__(self, tokenizer_cfg, memmap: memmap = None):
        super().__init__(tokenizer_cfg, memmap)
        self.meta: TransformersMetadata = TransformersMetadata(**asdict(self.meta), fast=self.tokenizer_cfg.get("fast", FAST))
        if self.tokenizer_cfg.get("fast") is None:
            log.warning(f'Missing key "fast" in transformers tokenizer config. Defaulting to True.')
            with open_dict(self.tokenizer_cfg):
                self.tokenizer_cfg.fast = FAST

        #get pretrainedtokenizer classes and their model encodings
        parent_class = transformers.PreTrainedTokenizerFast if self.meta.fast else transformers.PreTrainedTokenizer
        pretrained_tokenizers = get_classes(transformers.models, parent_class)
        self.tokenizer = None
        for tokenizer_cls_name, tokenizer_cls in pretrained_tokenizers.items():
            encodings = tokenizer_cls.max_model_input_sizes 
            if self.meta.encoding in encodings:
                self.tokenizer = tokenizer_cls.from_pretrained(self.meta.encoding, truncation=True)
        log.debug(f'Using tokenizer class: {self.tokenizer.__class__.__name__} with encoding: {self.meta.encoding}')
        if self.tokenizer is None:
            log.error(f'No transformers tokenizer found for encoding: {self.meta.encoding} and fast={self.meta.fast}')
            raise KeyError()
        self.tokenizer.model_max_length = 1e30 #disable warning about max context
        self.meta.max_token_value = self.tokenizer.vocab_size

    def encode(self, text) -> List[int]:
        #truncation is not performed as the tokens are appended into one continuous 1d memmap file
        #if truncation is performed, a large portion of the data is lost during tokenization
        tokens: list[int] = self.tokenizer(text)["input_ids"]
        self.meta.num_tokens += len(tokens)
        return tokens

    def decode(self, idx: List[int]) -> str:
        return self.tokenizer.decode(idx)