from typing import Any, Tuple, List
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig, open_dict
from qtransform.utils.introspection import get_classes, concat_paths, get_dtype
from qtransform.dataset import DatasetSplits, TokenizedDatasetGenerator
from qtransform.tokenizer import get_tokenizer, Tokenizer
import os
from glob import glob
import logging
from dataclasses import fields
import datasets
from datasets.dataset_dict import DatasetDict

class MemmapDataset(Dataset):
    
    def __init__(self, token_file: str, block_size: int, start: float=0.0, end: float = 1.0):
        """
            Creates a dataset which loads a one dimensional numpy array from a file. 
            
            Slices of the dataset can be retrieved with the start and end parameters. 
            They specify the starting and ending range of the dataset in percent.
            However, since each split is stored as one file, these two parameters have become obsolete.
        

        """
        super().__init__()
        if not isinstance(start, float) or start < 0.0 or start >= 1.0:
            log.error(f'Invalid starting range for dataset ({start})')
            raise KeyError()
        if not isinstance(end, float) or end <= 0.0 or end > 1.0:
            log.error(f'Invalid ending range for dataset ({end})')
            raise KeyError()
        if not isinstance(dtype, np.dtype): #np.dtype("dtype") or np.<dtype> e.g.: np.dtype("float32") or np.float32
            try:
                dtype = np.dtype(dtype) #not an instance (np.float32)
            except TypeError as e:
                log.error(e)
                raise TypeError
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
        self.data = np.memmap(self.token_file, dtype=self.dtype, mode='r', offset=offset)[:int(amnt_tokens * end)]
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
        if index < 0:
            index += len(self)
        #lower index to make sure that block_size elements are always retrieved
        if index + self.block_size > len(self) - 1:
            index = self.length - self.block_size - 2
        offset = index + self.block_size
        #fixed dtype as torch embeddings need int64 tensor to work
        #inputs: torch.Tensor = torch.from_numpy(self.data[index:offset].astype(np.int64))
        inputs: torch.Tensor = torch.as_tensor(self.data[index:offset].astype(np.int64))
        #labels are always the following word for each word within the context
        #labels : torch.Tensor = torch.from_numpy(self.data[index +1:offset+1].astype(np.int64))
        labels : torch.Tensor = torch.as_tensor(self.data[index +1:offset+1].astype(np.int64))
        return inputs, labels