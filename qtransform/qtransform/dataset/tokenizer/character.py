from glob import glob
from os import mkdir
from os.path import isdir, exists
import pickle
import numpy as np
from omegaconf import DictConfig
from qtransform.dataset.tokenizer import Tokenizer
from qtransform.utils.introspection import concat_paths
import logging

log = logging.getLogger(__name__)

class CharacterTokenizer(Tokenizer):

    def encode(stoi, s):
        return [stoi[c] for c in s] # encoder: take a string, output a list of integers
    def decode(itos, l):
        return ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

    def tokenize(tokenizer_cfg: DictConfig):
        log.debug(f'Tokenizing with parameters: {tokenizer_cfg}')
        root_path: list = tokenizer_cfg.dataset_dir #append directory seperator at the end of the path
        raw_dir = list()
        #pretty scuffed
        raw_dir.extend(root_path)
        raw_dir.extend(["untokenized", ""])
        raw_dir = concat_paths(raw_dir)
        log.debug(f'Checking for files under directory: {raw_dir}')
        chars = list()
        #iterate through each file in the untokenized directory, only include files at top level for now
        for file in [x for x in glob(raw_dir + tokenizer_cfg.name + '*') if not isdir(x)]:
            with open(file, 'r') as f:
                data = f.read()
            log.debug(f"length of dataset {file} in characters: {len(data)}")
            # get all the unique characters that occur in this text
            chars.extend(sorted(list(set(data))))
        # no files read?
        if len(chars) == 0:
            log.error(f'No readable files for tokenization at root level in {raw_dir} found')
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
        ids = np.array(CharacterTokenizer.encode(stoi, data), dtype=tokenizer_cfg.dtype)
        output_dir = list()
        output_dir.extend(root_path)
        output_dir.extend(["tokenized", ""])
        output_dir = concat_paths(output_dir)
        filename = tokenizer_cfg.name + "-" + tokenizer_cfg.encoding + "-" + tokenizer_cfg.dtype
        #directory seperator included in output_dir
        if not exists(output_dir):
            log.debug(f'Creating directory {output_dir}')
            mkdir(output_dir)
        with open(output_dir + filename + '-meta.pkl', 'wb') as f:
            pickle.dump(meta, f)
        ids.tofile(output_dir + filename + ".bin")