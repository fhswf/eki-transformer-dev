from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig
from qtransform.utils.introspection import get_classes
from qtransform.dataset import DatasetInfo, DatasetWrapper
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
        #e.g.: ~/.qtransform/cache/data/llm/tokenized/shakespeare/shakespeare-gpt2.bin
        # TODO find good structure for all our data
        root_path = os.path.join(data_cfg.root_path, "data", "llm", "tokenized", data_cfg.name, 
                        data_cfg.name + "-" + data_cfg.tokenizer.encoding + ".bin")
        #get dtype class to pass onto Dataset class
        dtype = None
        np_dtype_string = 'dtype['+data_cfg.dtype+']'
        for np_dtype in np.dtype.__subclasses__():
            if np_dtype_string == np_dtype.__name__:
                dtype = np_dtype
        if dtype == None:
            log.critical(f'Datatype {data_cfg.dtype} not found within numpy datatype scope')
            raise KeyError() 
        log.info(f'Loading dataset: {data_cfg.name}, with encoding: {data_cfg.tokenizer.encoding} and dtype: {data_cfg.dtype}')
        train = _FileSystemLLMDataset(root_path, dtype, data_cfg.block_size, end= 1.0 - data_cfg.args.split)
        test = _FileSystemLLMDataset(root_path, dtype, data_cfg.block_size, start= data_cfg.args.split)
        #transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)) ])
        return train, test

    
#TODO: implement using datasets from huggingface or pytorch
#TODO: implement download=True option
class _FileSystemLLMDataset(Dataset):
    
    def __init__(self, token_file: str, dtype: np.dtype, block_size: int, start: float=0.0, end: float = 1.0):
        """
            start: offset in dataset by <start> bytes
            end: end memmap by <end> entries of dtype
            For now, np.memmap is used with offset being start 
            and end being a slice of the memmap
        """
        super().__init__()
        #np.int32.b
        self.token_file = token_file
        self.dtype = dtype
        if not os.path.exists(token_file):
            log.warning(f"Tokenized dataset under \"{token_file}\" not found. Creating one...")
            pass
        #size of data in bytes, used for splitting training data
        size = os.path.getsize(token_file)
        #todo: make it somehow more efficient
        length_memmap = size * dtype.bit_count / 8
        self.data = np.memmap(token_file, dtype=dtype, mode='r', offset=int(start * length_memmap))[:int(length_memmap*end)]
        self.length = len(self.data)
        self.block_size = block_size
        
    def __len__(self):
        return self.length
    
    #the dataloader works with batch sizes, dataset only works with indices
    def __getitem__(self, index):
        # so you dont only get one token
        return self.data[index:index + self.block_size]
    
    def _gather_files(self, file_path: str):
        self.file_path = file_path
        file_list = glob.glob(self.file_path + "*")
        for file in file_list:
            pass        
        pass