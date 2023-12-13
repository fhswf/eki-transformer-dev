from numpy import array, memmap
from omegaconf import DictConfig
from qtransform.dataset.tokenizer import Tokenizer
import logging
from typing import List

log = logging.getLogger(__name__)

class CharacterTokenizer(Tokenizer):

    """
        Tokenizes one or multiple files inside of a directory into a singular binary file containing the numerical representation of
        each character accross all files. The id of each character is assigned iteratively (starting from zero) and put into 
        a numpy array. The data type is specified by the hydra config under data.dtype. It also creates a metadata file (.pkl) containing the
        vocabulary size and the corresponding mapping of characters to ids.
        The tokenization part is heavily inspired by nanoGPT (https://github.com/karpathy/nanoGPT/blob/master/data/shakespeare_char/prepare.py)
    """
    max_token_value = 2**16 -1 #Number of characters (tokens) depends on encoding. Python can have up to 4B large chars. Usually, 16 bits suffices
    stoi: dict = dict()
    itos: dict = dict()
    vocab_size: int = 0
    index_2d: int = 0
    vocab = list()

    def encode(s) -> List[int]:
        return [CharacterTokenizer.stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(l) -> str:
        return ''.join([CharacterTokenizer.itos[i] for i in l]) # decoder: take a list of integers, output a string
    
    def flush_vocab():
        #maybe the tokenizers should have a constructor
        CharacterTokenizer.stoi = dict()
        CharacterTokenizer.itos = dict()
        CharacterTokenizer.vocab_size = 0
    
    def tokenize(memmap: memmap, text: str, tokenizer_cfg: DictConfig):
        """
            The function the steps described in the class description (tokenization of files into one binary file). It is usually
            called by the tokenizer module in order to abstract the tokenization process.
        """
        num_items, num_rows = Tokenizer.get_memmap_dimension(memmap)
        log.critical("ALKJDHAKSJDHKJLASDH")
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        #python chars are 2B large, meaning that at the absolute most only
        #2^16 tokens are created (65.000) if every single unicode character appears in dataset
        CharacterTokenizer.vocab = list()
        #iterate through each file in the untokenized directory, only include files at top level for now
        #untokenized_files = get_files(tokenizer_cfg)
        #only update vocab with new characters
        new_tokens = set(text) - set(CharacterTokenizer.vocab) 
        CharacterTokenizer.vocab.extend(new_tokens)
        #remember the token of previous characters, start counting at vocab_size
        CharacterTokenizer.stoi.update({ ch:i for i,ch in enumerate(new_tokens, CharacterTokenizer.vocab_size) })
        CharacterTokenizer.itos.update({ i:ch for i,ch in enumerate(new_tokens, CharacterTokenizer.vocab_size) })
        CharacterTokenizer.vocab_size += len(new_tokens)
        if CharacterTokenizer.vocab_size > 2 ** memmap.dtype.itemsize * 8 -1:
            log.error(f'Vocab size is larger than what the memmap can store ({memmap.dtype})')
            raise TypeError() 
        if num_rows == 1: #memmap is one large 1d array, simply append tokens
            memmap[CharacterTokenizer.vocab_size -1 : CharacterTokenizer.vocab_size -1 + len(text)] = CharacterTokenizer.encode(text)
        else:   #2d array, remember index to write into
            memmap[index_2d] = CharacterTokenizer.encode(text)
            index_2d += 1
        #save changes
        memmap.flush()

    def save_metadata():
        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': CharacterTokenizer.vocab_size,
            'itos': itos,
            'stoi': stoi,
        }