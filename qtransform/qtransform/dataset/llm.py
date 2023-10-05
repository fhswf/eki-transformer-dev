from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig
from qtransform.utils.introspection import get_classes, concat_paths, get_dtype
from qtransform.dataset import DatasetInfo, DatasetWrapper
from qtransform.dataset.tokenizer import get_tokenizer, Tokenizer
import os
import glob
import logging
log = logging.getLogger(__name__)

class FileSystemLLMDataset(DatasetInfo, DatasetWrapper):
    """
        DatasetWrapper used to load .bin files from root_path and return a Dataset storing a numpy memmap
    """
    def __init__(self) -> None:
        pass

    def load_dataset(data_cfg: DictConfig) -> Dataset:
        # TODO find good structure for all our data
        paths: list = data_cfg.dataset_dir
        paths.extend(["tokenized", data_cfg.name + "-" + data_cfg.tokenizer.encoding + "-" + data_cfg.dtype + ".bin"])
        root_path = concat_paths(paths)
        #get dtype class to pass onto Dataset class
        dtype = get_dtype(data_cfg.dtype)
        log.info(f'Loading dataset: {data_cfg.name}, with encoding: {data_cfg.tokenizer.encoding} and dtype: {data_cfg.dtype}')
        if not os.path.exists(root_path):
            tokenizer: Tokenizer = get_tokenizer(data_cfg.tokenizer)
            tokenizer.tokenize(data_cfg.tokenizer)
        split = data_cfg.args.split
        if split == None:
            split = 0.0
        train = _FileSystemLLMDataset(root_path, dtype, data_cfg.block_size, end= 1.0 - split)
        test = _FileSystemLLMDataset(root_path, dtype, data_cfg.block_size, start= split)
        #transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)) ])
        return train, test

    
#TODO: implement using datasets from huggingface or pytorch
#TODO: implement download=True option
class _FileSystemLLMDataset(Dataset):
    
    def __init__(self, token_file: str, dtype: np.dtype, block_size: int, start: float=0.0, end: float = 1.0):
        """
            The seperation of a dataset in training and validation splits is done with the start and end parameter

            start: offset in dataset by <start> bytes (start counting after the first start percent items)
            end: end memmap by <end> entries of dtype
            For now, np.memmap is used with offset being start 
            and end being a slice of the memmap
        """
        super().__init__()
        log.info(f"Attempting to retrieve tokenized dataset under \"{token_file}\"")
        self.token_file = token_file
        self.dtype = dtype
        self.datatype = dtype
        #the method of retrieving the byte size is somewhat inspired from the stackoverflow article
        #https://stackoverflow.com/questions/19599864/easy-way-of-getting-number-of-bits-from-a-numpy-type
        self.bytes = self.datatype.itemsize
        amnt_tokens = os.path.getsize(self.token_file) / self.bytes
        if amnt_tokens % 1 != 0.0:
            log.error(f'The amount of tokens is supposed to be a whole number, but it is {amnt_tokens}. Maybe due to a wrong datatype?')
            raise ValueError()
        offset = int(amnt_tokens * start)
        #rounding to the nearest multiplicative of datatype to make sure not to read half a token too much
        offset -= offset % self.bytes
        #skip the first start * amnt_tokens and the last amnt_tokens * end items
        log.debug(f'Tokenized file has {amnt_tokens} tokens of datatype: {dtype}. Attempting to start at token: {offset}')
        self.data = np.memmap(self.token_file, dtype=self.datatype, mode='r', offset=offset)[:int(amnt_tokens * end)]
        log.debug(f'memmap has been created with dtype: {dtype}')
        self.length = len(self.data)
        self.block_size = block_size
        
    def __len__(self):
        return self.length
    
    #the dataloader works with batch sizes, dataset only works with indices
    def __getitem__(self, index):
        #From https://pytorch.org/docs/stable/generated/torch.from_numpy.html:
        #The returned tensor and ndarray share the same memory. Modifications to the tensor will be reflected in the ndarray and vice versa. The returned tensor is not resizable.
        #therefore, copy part of np array or use torch.stack()
        return torch.from_numpy(np.copy(self.data[index:index + self.block_size]))
    
    def _gather_files(self, file_path: str):
        self.file_path = file_path
        file_list = glob.glob(self.file_path + "*")
        for file in file_list:
            pass        
        pass