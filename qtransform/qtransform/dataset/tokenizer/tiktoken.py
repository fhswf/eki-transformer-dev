from typing import List
from tiktoken import get_encoding, Encoding
from qtransform.dataset.tokenizer import Tokenizer
from omegaconf import DictConfig
import logging
from numpy import array, dtype

log = logging.getLogger(__name__)

class TikTokenizer(Tokenizer):
    """
        Uses tiktoken as the tokenizer of choice. It calls encode_ordinary with an object
        containing the specified encoding, e.g. gpt2.
    """
    def tokenize(tokenizer_cfg: DictConfig):
        """
            Tokenize an input from file under text_file and write the generated bin file in root_path/root_path-<encoding>.bin. 
            Since BPE is used, it is possible that the value of tokens cannot be supported by the datatype due to insufficient amount of bits.
        """
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        try:
            encoder: Encoding = get_encoding(tokenizer_cfg.encoding)
        except ValueError as e:
            log.error(f'Could not load Tiktoken tokenizer with encoding: "{tokenizer_cfg.encoding}".')
            raise ValueError()
        tokens: List[int] = list()
        #int64: 2^(8*8)
        #TODO: move this into llm and pass dataset as argument
        highest_token_allowed = 2**(dtype(tokenizer_cfg.dtype).itemsize*8) - 1 
        for file in get_files(tokenizer_cfg):
            with open(file, 'r') as f:
                try:
                    data = f.read()
                    data_ids = encoder.encode_ordinary(data)
                    highest_token = max(data_ids)
                    if highest_token > highest_token_allowed:
                        log.error(f"Tokenizing the dataset with encoding: {tokenizer_cfg.encoding} and dtype: {tokenizer_cfg.dtype} will result in a conversion loss\n\
                                    (Highest token in prepared dataset: {highest_token}, highest allowed token due to dtype: {highest_token_allowed})")
                        raise ValueError()
                    log.debug(f"Amount of tokens in dataset {file}: {len(data_ids)}")
                    tokens.extend(data_ids)
                except PermissionError:
                    log.warning(f'Could not read file "{file}" for tokenization.')
        ids = array(tokens, dtype=tokenizer_cfg.dtype)
        save_tokens(ids, tokenizer_cfg)