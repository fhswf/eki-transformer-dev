from typing import Any, Tuple, List
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig
from qtransform.utils.introspection import get_classes, concat_paths, get_dtype
from qtransform.dataset import DatasetInfo, DatasetWrapper
from qtransform.dataset.tokenizer import get_tokenizer, Tokenizer
import os
from glob import glob
import logging
from dataclasses import fields

log = logging.getLogger(__name__)

class FileSystemLLMDatasetWrapper(DatasetWrapper):
    """
        DatasetWrapper used to load .bin files from a dataset file and return a Dataset storing a numpy memmap
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        if self.cfg.args.get('dtype') is None:
            log.error(f'Missing dtype for "{self.cfg.name}" dataset.')
            raise KeyError()
        #currently, dtype has to be set by user. maybe it could also be automatically infered by the max tokens property of Tokenizer
        self.dtype = get_dtype(self.cfg.args.dtype)
        #directories for untokenized and tokenized files
        self.untokenized_dir = concat_paths([*cfg.dataset_dir, "untokenized"])
        self.tokenized_dir = concat_paths([*cfg.dataset_dir, "tokenized", cfg.tokenizer.encoding])
        self.dataset_file = os.path.join(self.tokenized_dir, self.cfg.name+ '-' + self.cfg.tokenizer.dtype + '.bin')

    def load_dataset(self):
        log.info(f'Loading dataset: {self.cfg.name}, with encoding: {self.cfg.tokenizer.encoding} and dtype: {self.dtype}')
        #check if tokenized file exists. if not, create it
        if not os.path.exists(self.dataset_file):
            os.makedirs(self.tokenized_dir, exist_ok = True)
            #no instance, only classname
            files = self.get_untokenized_files()
            #get length of characters of each file in order to specify the shape of memmap
            #to do that, each file has to be read twice which is pretty inefficient
            #maybe consider: https://numpy.org/doc/stable/reference/generated/numpy.memmap.reshape.html
            amount_characters = 0
            for file in files:
                with open(file, 'r') as text_file: 
                    chars_file = 0
                    #should newline seperators (\n) be removed from tokens?
                    for line in text_file:
                        chars_file += len(line)  
                    amount_characters += chars_file
                    log.debug(f'File {file} has {chars_file} characters.')
                    
            memmap = np.memmap(self.dataset_file, dtype=self.dtype, mode='w+', shape=(amount_characters))
            tokenizer: Tokenizer = get_tokenizer(self.cfg.tokenizer, memmap=memmap)
            #actually tokenize files, line by line
            for file in files:
                log.debug(f'Tokenizing file: {file} with encoding: {self.cfg.tokenizer.encoding}')
                with open(file, 'r') as text_file: 
                    for line in text_file:
                        tokenizer.tokenize(line)
            log.debug(f'Vocab size: {tokenizer.vocab_size}. Number of tokens: ')
            memmap.flush()
            tokenizer.save_metadata(self.tokenized_dir)
        #train
        if self.dataset_sizes.train > 0.0:
            self.dataset_info.train = _FileSystemLLMDataset(self.dataset_file, self.dtype, self.cfg.args.block_size, end=self.dataset_sizes.train)
        #eval
        if self.dataset_sizes.eval > 0.0:
            percentage_eval = round(self.dataset_sizes.eval * 100)
            eval_start = torch.randint(round(self.dataset_sizes.train * 100) - percentage_eval, (1, )).item() / 100
            self.dataset_info.eval = _FileSystemLLMDataset(self.dataset_file, self.dtype, self.cfg.args.block_size, start=eval_start, end=eval_start + self.dataset_sizes.eval)
        #bench
        if self.dataset_sizes.bench > 0.0:
            self.dataset_info.test = _FileSystemLLMDataset(self.dataset_file, self.dtype, self.cfg.args.block_size, end=self.dataset_sizes.test)
        #test
        if self.dataset_sizes.test > 0.0:
            #for now, use last percent of dataset for testing
            self.dataset_info.test = _FileSystemLLMDataset(self.dataset_file, self.dtype, self.cfg.args.block_size, start= 1.0 - self.dataset_sizes.test)

    def get_untokenized_files(self) -> List:
        """
            Returns all readable files from a given directory which are going to be used for tokenization. 
            To do so, the field "dataset_dir" from the hydra config is evaluated. All files from the directory "untokenized"
            within dataset_dir are returned. 
            If the directory does not exist, it is created and an empty list is returned.
            
        """
        main_path = concat_paths([*self.cfg.dataset_dir, "untokenized", ""])
        if not os.path.exists(main_path):
            log.debug(f'Creating directory {main_path}')
            os.makedirs(main_path, exist_ok=True)
            return []
        log.debug(f'Checking for files with name containing {self.cfg.name} under directory: {main_path}')
        return [x for x in glob(main_path + self.cfg.name + '*') if not os.path.isdir(x)]

    def shuffle(self):
        raise NotImplementedError()

#TODO: implement download=True option
class _FileSystemLLMDataset(Dataset):
    
    #TODO: is dtype necessary? since each file only contains the ids of tokens, not the actual embeddings
    def __init__(self, token_file: str, dtype: np.dtype, block_size: int, start: float=0.0, end: float = 1.0):
        """
            Creates a dataset which loads a numpy array from a file. 
            Slices of the dataset can be retrieved with the start and end parameters. 
            They specify the starting and ending range of the dataset in percent.
        """
        super().__init__()
        if not isinstance(start, float) or start < 0.0 or start >= 1.0:
            log.error(f'Invalid starting range for dataset ({start})')
            raise KeyError()
        if not isinstance(end, float) or end <= 0.0 or end > 1.0:
            log.error(f'Invalid ending range for dataset ({end})')
            raise KeyError()
        self.block_size = block_size
        if self.block_size <= 0:
            log.error(f'Block size of 0 is invalid.')
            raise ValueError()
        log.info(f"Attempting to retrieve tokenized dataset under \"{token_file}\"")
        self.token_file = token_file
        self.dtype = dtype
        #the method of retrieving the byte size is somewhat inspired from the stackoverflow article
        #https://stackoverflow.com/questions/19599864/easy-way-of-getting-number-of-bits-from-a-numpy-type
        self.bytes = self.dtype.itemsize
        amnt_tokens = os.path.getsize(self.token_file) / self.bytes
        if amnt_tokens % 1 != 0.0:
            log.error(f'The amount of tokens is supposed to be a whole number, but it is {amnt_tokens}. Maybe due to a wrong datatype?')
            raise ValueError()
        offset = int(amnt_tokens * start)
        #rounding to the nearest multiplicative of datatype to make sure not to read half a token too much
        offset -= offset % self.bytes
        log.debug(f'Offset is {offset}, start is {start}, end is {end}')
        #skip the first start * amnt_tokens and the last amnt_tokens * end items
        log.debug(f'Tokenized file has {amnt_tokens} tokens of datatype: {dtype}. Attempting to start at token: {offset}')
        #torch.nn.Embedding only takes inputs of type int (32b, 64b) -> cast np array to 64 signed int
        self.data = np.memmap(self.token_file, dtype=self.dtype, mode='r', offset=offset)[:int(amnt_tokens * end)].astype(np.int64)
        if len(self.data) < self.block_size:
            log.error(f'Loaded data has less tokens than block size {self.block_size} for starting range {start} and ending range {end}. Maybe check size of splits?')
            raise ValueError()
        log.info(f'Loaded data has {len(self.data)} tokens.')
        #log.debug(f'all unique tokens in dataset: {set(self.data)}, length: {len(set(self.data))}')
        self.length = len(self.data)

    def __len__(self):
        return self.length
    
    #the dataloader works with batch sizes, dataset only works with indices
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Returns inputs and labels. Called when iterating with e.g. a dataloader.
        """
        #assert index >= 0
        #lower index to make sure that block_size elements are always retrieved
        index = min(self.length - self.block_size - 2, index + self.block_size)
        offset = index + self.block_size
        #From https://pytorch.org/docs/stable/generated/torch.from_numpy.html:
        #The returned tensor and ndarray share the same memory. Modifications to the tensor will be reflected in the ndarray and vice versa. 
        #The returned tensor is not resizable.
        #therefore, copy part of np array or use torch.stack()
        inputs: torch.Tensor = torch.from_numpy(np.copy(self.data[index:offset]))
        #labels are always the following word for each word within the context
        labels : torch.Tensor = torch.from_numpy(np.copy(self.data[index +1:offset+1]))
        return inputs, labels