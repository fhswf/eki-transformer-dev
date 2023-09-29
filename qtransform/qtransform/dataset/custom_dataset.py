from typing import Any
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig
from qtransform.utils.introspection import get_classes
from qtransform.dataset import DatasetInfo, DatasetWrapper
import os
import glob

class FileSystemLLMDataset(DatasetInfo, DatasetWrapper):
    """
        DatasetWrapper used to load .bin files from root_path and return a Dataset storing a numpy memmap
    """
    def __init__(self) -> None:
        pass

    def load_dataset(data_cfg: DictConfig) -> Dataset:
        pass
        # TODO find good structure for all our data
        #e.g.: ~/.qtransform/cache/data/llm/tokenized/shakespeare/shakespeare-gpt2.bin
        root_path = os.path.join(data_cfg.root_path, "data", "llm", "tokenized", data_cfg.name, 
            data_cfg.name + "-" + data_cfg.encoding + ".bin")
        dtype = get_classes('', np.dtype )
        dataset = _FileSystemLLMDataset(root_path, dtype, data_cfg.block_size )

        transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)) ])
        train = available_datasets[cfg.name](root=root_path, train=True, download=True, transform=transform)
        test = available_datasets[cfg.name](root=root_path, train=False, transform=transform)
        return train, test

    
#TODO: implement using datasets from huggingface or pytorch
#TODO: implement download=True option
class _FileSystemLLMDataset(torch.utils.data.Dataset):
    
    def __init__(self, token_file: str, dtype, block_size, download = True, split = 0):
        super().__init__()
        np.int32.b
        self.token_file = token_file
        self.dtype = dtype
        if download and not os.path.exists(token_file):
            #download file from datasets.json
            ""
        #used for splitting training data
        size = os.path.getsize(token_file)
        #todo: make it somehow more efficient
        self.data = np.memmap(token_file, dtype=dtype, mode='r', offset=size*split*dtype.bit_count / 8)
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