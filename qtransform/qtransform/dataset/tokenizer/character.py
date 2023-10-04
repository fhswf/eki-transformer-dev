import os
import pickle
import numpy as np
from omegaconf import DictConfig
import torch
from qtransform.dataset.tokenizer import Tokenizer
from qtransform.utils.introspection import concat_paths
import logging

log = logging.getLogger(__name__)

class CharacterTokenizer(Tokenizer):

    def encode(stoi, s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(itos, l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    #files: list, dtype: np.dtype
    def tokenize(tokenizer_cfg: DictConfig):
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        root_path: list = tokenizer_cfg.dataset_dir.extend("") #append directory seperator at the end of the path 
        raw_dir = root_path
        raw_dir = concat_paths(raw_dir.extend(["untokenized"]))
        chars = list()
        #iterate through each file in the untokenized directory, only include files at top level for now
        for file in os.walk(raw_dir)[0][2]:
            with open(file, 'r') as f:
                data = f.read()
            log.debug(f"length of dataset {file} in characters: {len(data):,}")
            # get all the unique characters that occur in this text
            chars.extend(sorted(list(set(data))))
        # no files read?
        if len(chars) == 0:
            log.error(f'No readable files for tokenization in {raw_dir} found')
            raise KeyError()
        vocab_size = len(chars)
        log.debug("all the unique characters:", ''.join(chars))
        log.debug(f"vocab size: {vocab_size:,}")
        # create a mapping from characters to integers
        stoi = { ch:i for i,ch in enumerate(chars) }
        itos = { i:ch for i,ch in enumerate(chars) }
        # save the meta information as well, to help us encode/decode later
        meta = {
            'vocab_size': vocab_size,
            'itos': itos,
            'stoi': stoi,
        }
        ids = np.array(CharacterTokenizer.encode(stoi, data), dtype=tokenizer_cfg.dtype)
        output_dir = root_path
        output_dir = concat_paths(output_dir.extend(["tokenized", ""]))
        filename = tokenizer_cfg.name + "-" + tokenizer_cfg.tokenizer.encoding + "-" + tokenizer_cfg.dtype
        #directory seperator included in output_dir
        with open(output_dir + filename + 'meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
        ids.tofile(output_dir + filename + ".bin")