from typing import Any, Tuple
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
from dataclasses import fields

log = logging.getLogger(__name__)

class FileSystemLLMDatasetWrapper(DatasetWrapper):
    """
        DatasetWrapper used to load .bin files from root_path and return a Dataset storing a numpy memmap
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        if self.cfg.args.get('dtype') is None:
            log.error(f'Missing dtype for "{self.cfg.name}" dataset.')
            raise KeyError()
        self.dtype = get_dtype(self.cfg.args.dtype)
        # TODO find good structure for all our data
        paths: list = self.cfg.dataset_dir
        paths.extend(["tokenized", self.cfg.name + "-" + self.cfg.tokenizer.encoding + "-" + self.cfg.args.dtype + ".bin"])
        self.root_path = concat_paths(paths) ## TODO replace all "~" in conf

    def load_dataset(self, split: str) -> DatasetInfo:
        self.check_split(split)
        if getattr(self.dataset_sizes, split) == 0.0:
            log.error(f'Cannot load dataset for split {split} since it was configured to be empty.')
            raise ValueError()
        log.info(f'Loading dataset: {self.cfg.name}, with encoding: {self.cfg.tokenizer.encoding} and dtype: {self.dtype}')
        if not os.path.exists(self.root_path):
            #no instance, only classname
            tokenizer: Tokenizer = get_tokenizer(self.cfg.tokenizer)
            tokenizer.tokenize(self.cfg.tokenizer)
        dataset_info = DatasetInfo(name = self.cfg.name)
        #instead of specifying a split, create splits based on size
        #if split is equal to 0.0 (default), leave split empty (None)
        #TODO: find out if this could lead to storage/ memory issues if datasets are large
        """ 
        for split in [x.name for x in fields(self.dataset_sizes) if x.name not in ["train", "eval"]]:
            split_size = getattr(self.dataset_sizes, split)
            if split_size <= 0.0:
                continue
            setattr(dataset_info, split, _FileSystemLLMDataset(self.root_path, self.dtype, self.cfg.args.block_size, end=))
        #train and eval splits are created seperately as eval needs to be within training split
        if self.dataset_sizes.train > 0.0:
            dataset_info.train = _FileSystemLLMDataset(self.root_path, self.dtype, self.cfg.args.block_size, end=self.dataset_sizes.train)
            percentage_eval = round(self.dataset_sizes.eval * 100)
        if self.dataset_sizes.eval > 0.0:
            eval_start = torch.randint(round(self.dataset_sizes.train * 100) - percentage_eval, (1, )).item() / 100
            dataset_info.eval = _FileSystemLLMDataset(self.root_path, self.dtype, self.cfg.args.block_size, start=eval_start, end=eval_start + self.dataset_sizes.eval)
        """
        match split:
            case 'train':
                dataset_info.train = _FileSystemLLMDataset(self.root_path, self.dtype, self.cfg.args.block_size, end=self.dataset_sizes.train)
                if self.dataset_sizes.eval > 0.0:
                    percentage_eval = round(self.dataset_sizes.eval * 100)
                    eval_start = torch.randint(round(self.dataset_sizes.train * 100) - percentage_eval, (1, )).item() / 100
                    dataset_info.eval = _FileSystemLLMDataset(self.root_path, self.dtype, self.cfg.args.block_size, start=eval_start, end=eval_start + self.dataset_sizes.eval)
            case 'bench':
                dataset_info.test = _FileSystemLLMDataset(self.root_path, self.dtype, self.cfg.args.block_size, end=self.dataset_sizes.test)
            case 'test':
                #for now, use last percent of dataset for testing
                dataset_info.test = _FileSystemLLMDataset(self.root_path, self.dtype, self.cfg.args.block_size, start= 1.0 - self.dataset_sizes.test)
        return dataset_info
    
    def shuffle(self):
        raise NotImplementedError()

#TODO: implement download=True option
class _FileSystemLLMDataset(Dataset):
    
    def __init__(self, token_file: str, dtype: np.dtype, block_size: int, start: float=0.0, end: float = 1.0):
        """
            The seperation of a dataset in different splits is done with the start and end parameter

            start: offset in dataset by <start> bytes (start counting after the first start percent items)
            end: end memmap by <end> entries of dtype
            For now, np.memmap is used with offset being start 
            and end being a slice of the memmap
        """
        super().__init__()
        if not isinstance(start, float) or start < 0.0 or start >= 1.0:
            log.error(f'Invalid starting range for dataset ({start})')
            raise KeyError()
        if not isinstance(end, float) or end <= 0.0 or end > 1.0:
            log.error(f'Invalid ending range for dataset ({end})')
            raise KeyError()
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
        #log.debug(f'all unique tokens in dataset: {set(self.data)}, length: {len(set(self.data))}')
        self.length = len(self.data)
        self.block_size = block_size
        if self.block_size == 0:
            log.error(f'Block size of 0 is invalid.')
            raise ValueError()
        if self.length <= self.block_size - 2 :
            log.warn(f'Data samples are always going to be the same as the block size ({self.block_size}) is greater or equal to the dataset length. Setting block_size to: {self.length - 2}')
            self.block_size = self.length - 2

    def __len__(self):
        return self.length
    
    #the dataloader works with batch sizes, dataset only works with indices
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Returns inputs and labels. Called when iterating through a dataloader.
        """
        #assert index >= 0
        #make sure that block_size elements are always retrieved
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