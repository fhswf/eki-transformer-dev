from typing import List, Dict, Any
from qtransform.tokenizer import Tokenizer, Metadata
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

FAST = False #by default, use fast implementation of tokenizers

@dataclass
class TransformersMetadata(Metadata):
    module: str = "transformers"
    fast: bool = FAST

log = logging.getLogger(__name__)

#TODO: make use of AutoTokenizer
class TransformersTokenizer(Tokenizer):
    """
        Uses transformers as the tokenizer of choice.
        You can choose between the "full" version or the fast version. Differences can be read on 
        https://huggingface.co/docs/transformers/main_classes/tokenizer#tokenizer (from 13.12.2023)
    """
    def __init__(self, tokenizer_cfg):
        super().__init__(tokenizer_cfg)
        self.meta: TransformersMetadata = TransformersMetadata(**asdict(self.meta), fast=self.tokenizer_cfg.get("fast", FAST))
        if self.tokenizer_cfg.get("fast") is None:
            log.warning(f'Missing key "fast" in transformers tokenizer config. Defaulting to False.')
            with open_dict(self.tokenizer_cfg):
                self.tokenizer_cfg.fast = FAST
        #support third party tokenizers
        if "pretrained_tokenizer" in self.tokenizer_cfg and self.tokenizer_cfg.pretrained_tokenizer is not None:
            tokenizer_cls = getattr(transformers, self.tokenizer_cfg.pretrained_tokenizer, None)
            assert tokenizer_cls is not None, \
                f'Pretrained Tokenizer {self.tokenizer_cfg.pretrained_tokenizer} does not exist'
            self.tokenizer = tokenizer_cls.from_pretrained(self.meta.encoding, truncation=True)
        else: 
            #no specific pretrainedtokenizer set, find suitable one from AutoTokenizer
            try:
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.tokenizer_cfg.encoding)
            except Exception as e:
                log.error(f'Could not find tokenizer with encoding: "{self.tokenizer_cfg.encoding}"', exc_info=True)
                raise e
        log.debug(f'Using tokenizer class: {self.tokenizer.__class__.__name__} with encoding: {self.meta.encoding}')
        if self.tokenizer is None:
            log.error(f'No transformers tokenizer found for encoding: {self.meta.encoding} and fast={self.meta.fast}. Maybe manually set pretrained_tokenizer')
            raise KeyError()
        #since model is only needed to tokenize dataset, context length does not matter
        #increasing context length gets rid of warning
        self.tokenizer.model_max_length = 1e30
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.deprecation_warnings["Asking-to-pad-a-fast-tokenizer"] = True
        #https://github.com/huggingface/datasets/issues/3638
        self.tokenizer("Call init Tokenier", "to enable cacheing bug", truncation=True)
        self.meta.max_token_value = self.tokenizer.vocab_size

    def encode(self, text, infer: bool = False) -> List[int]:
        #truncation is not performed as the tokens are appended into one continuous 1d memmap file
        #if truncation is performed, a large portion of the data is lost during tokenization
        tokens: list[int] = self.tokenizer(text)["input_ids"]
        self.meta.num_tokens += len(tokens) if not infer else 0
        return tokens

    def decode(self, idx: List[int]) -> str:
        return self.tokenizer.decode(idx)