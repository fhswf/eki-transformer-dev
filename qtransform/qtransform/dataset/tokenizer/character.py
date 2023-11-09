from numpy import array
from omegaconf import DictConfig
from qtransform.dataset.tokenizer import Tokenizer, get_files, save_tokens
import logging

log = logging.getLogger(__name__)

class CharacterTokenizer(Tokenizer):

    """
        Tokenizes one or multiple files inside of a directory into a singular binary file containing the numerical representation of
        each character accross all files. The id of each character is assigned iteratively (starting from zero) and put into 
        a numpy array. The data type is specified by the hydra config under data.dtype. It also creates a metadata file (.pkl) containing the
        vocabulary size and the corresponding mapping of characters to ids.
        The tokenization part is heavily inspired by nanoGPT (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py)
    """
    def encode(stoi, s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(itos, l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    def tokenize(tokenizer_cfg: DictConfig):
        """
            The function the steps described in the class description (tokenization of files into one binary file). It is usually
            called by the tokenizer module in order to abstract the tokenization process.
        """
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        chars = list()
        #iterate through each file in the untokenized directory, only include files at top level for now
        untokenized_files = get_files(tokenizer_cfg)
        log.debug(f'Found files: {untokenized_files}')
        for file in untokenized_files:
            with open(file, 'r') as f:
                try:
                    data = f.read()
                    log.debug(f"length of dataset {file} in characters: {len(data)}")
                    # get all the unique characters that occur in this text
                    chars.extend(sorted(list(set(data))))
                except PermissionError:
                    pass
        # no files read?
        if len(chars) == 0:
            log.error(f'No readable files for tokenization found')
            raise KeyError()
        vocab_size = len(chars)
        #log.debug("all the unique characters:", ''.join(chars))
        #log.debug(f"vocab size: {vocab_size:,}")
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        ids = array(CharacterTokenizer.encode(stoi, data), dtype=tokenizer_cfg.dtype)
        #ids: ndarray,tokenizer_cfg: DictConfig, meta: Dict = None
        save_tokens(ids, tokenizer_cfg, meta)