from typing import List
from qtransform.dataset.tokenizer import Tokenizer
from omegaconf import DictConfig
import logging
import transformers
from datasets import Dataset#, PreTrainedTokenizer, PreTrainedTokenizerFast
from qtransform.classloader import get_data

log = logging.getLogger(__name__)

class TransformersTokenizer(Tokenizer):
    """
        Uses transformers as the tokenizer of choice.
        You can choose between the "full" version or the fast version. Differences can be read on 
        https://huggingface.co/docs/transformers/main_classes/tokenizer#tokenizer (from 13.12.2023)
    """
    def tokenize(tokenizer_cfg: DictConfig, hf_dataset: Dataset):
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        #TODO: transformers tokenization
        ids = array(tokens, dtype=tokenizer_cfg.dtype)
        save_tokens(ids, tokenizer_cfg)
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

