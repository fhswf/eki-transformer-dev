from typing import List
from qtransform.dataset.tokenizer import Tokenizer
from numpy import memmap
from omegaconf import DictConfig, open_dict
import logging
import transformers
from datasets import Dataset#, PreTrainedTokenizer, PreTrainedTokenizerFast
from qtransform.classloader import get_data
from re import match, search

log = logging.getLogger(__name__)
TOKENIZERS = {x.__name__: x for x in transformers.PreTrainedTokenizer.__subclasses__()}

class TransformersTokenizer(Tokenizer):
    """
        Uses transformers as the tokenizer of choice.
        You can choose between the "full" version or the fast version. Differences can be read on 
        https://huggingface.co/docs/transformers/main_classes/tokenizer#tokenizer (from 13.12.2023)
    """
    def tokenize(tokenizer_cfg: DictConfig, memmap: memmap):
        super().__init__(tokenizer_cfg = tokenizer_cfg, memmap=memmap)
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        #gpt2 should become GPT2Tokenizer(Fast)
        encoding = filter(lambda x: x.lower(), search(r'^(.+)Tokenizer(Fast){0,1}$', self.tokenizer_cfg.encoding).groups)
        if len(encoding) > 1:
            log.warning(f'Encoding should not contain the Tokenizer class. Extracting encoding: {encoding[0]}')
            with open_dict(self.tokenizer_cfg):
                self.tokenizer_cfg.encoding = encoding[0]
        self.tokenizer = None
        for tokenizer_name, tokenizer_class in TOKENIZERS.items():
            #TODO: find correct Tokenizer class
            pass
        if self.tokenizer is None:
            log.error(f'No transformers tokenizer found for encoding: {self.tokenizer_cfg.encoding} and fast={self.tokenizer_cfg.fast}')
            raise KeyError()
        self.tokenizer = self.tokenizer.from_pretrained(self.tokenizer_cfg.encoding)
        """

        #tokenizers return a dictionary containing the input ids and the attention mask
        #attention mask basically is a list of ones the length of the ids
        #TODO: return list of ints instead of dict for compatibility
        #tokenizer = AutoTokenizer.from_pretrained("gpt2",kwargs={"max_length": 1024})
        # TODO cfg this
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        #parameter is dataset containing text and other properties. only text is important
        def tokenization(example):
            # TODO cfg this
            #max_length is the length of the attention mask
            #is attention mask necessary?
            return tokenizer(example["text"], max_length=1024, truncation=True)
        """
    def tokenize(self, text) -> List[int]:
        tokens: list[int] = list()
        #TODO: consider max_model_input_sizes (e.g. only 1024 words at a time for gpt2)
        self.num_tokens += len(tokens)
        return tokens
