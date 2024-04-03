from typing import Any, Tuple, List
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig, open_dict
from qtransform.utils.introspection import get_classes, concat_paths, get_dtype
from qtransform.dataset import DatasetSplits, TokenizedDatasetGenerator, DatasetSplitType
from qtransform.tokenizer import get_tokenizer, Tokenizer
import os
from glob import glob
import logging
from dataclasses import fields
import datasets
from datasets.dataset_dict import DatasetDict
from typing import Any, Union, Dict, Callable, Tuple, List
from omegaconf import DictConfig, open_dict
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes, concat_paths, get_dtype
import qtransform
from qtransform import classloader
from dataclasses import dataclass, fields, InitVar
from enum import IntEnum
from os import listdir, makedirs
from os.path import join, exists
from qtransform.tokenizer import Tokenizer, get_tokenizer
import datasets
from datasets.dataset_dict import DatasetDict
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
import qtransform.dataset as package_self
from transformers import DataCollatorForLanguageModeling, DataCollatorWithPadding
from qtransform import device_singleton
#TokenizedDatasetWrapper
import numpy as np
from typing import Tuple
import torch
from pprint import PrettyPrinter


class FilesTokenizedDatasetGenerator(TokenizedDatasetGenerator):
    """
    TokenizedDatasetGenerator used to load huggingface datasets and tokenize datasets into arrow files.
    """
    #contains file extension and other distinguishing factors (e.g. block_size, tokenized or grouped..)
    #split is prepended to suffix (cache_file_prefix, split, DATASET_FILE_SUFFIX)
    DATASET_FILE_SUFFIX: str
    DATASET_FILE_PATH: str #by default: cache_dir from config

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.DATASET_FILE_SUFFIX = ".bin"
        self.RAW_DATA_DIR = self.cfg.untokenized.data_dir
        log.debug(f'{self.RAW_DATA_DIR}')
        #make __post_init__?
        makedirs(self.DATASET_FILE_PATH, exist_ok=True)
        self.batch_size = self.cfg.untokenized.args.batches
        if self.batch_size is None:
            self.batch_size = 1000 #default for huggingface

    def get_tokenized_dataset(self) -> DatasetSplits:
        dataset_splits = DatasetSplits()
        for split in DatasetSplitType:
            try:
                dataset_splits[split] = np.memmap(self.get_filepath_split(split), mode='r')
            except FileNotFoundError as e:
                log.error(f'Split {split.name} not found locally.')
                raise e
        return dataset_splits

    def chunk_examples(self, examples):
        """
            Splits the text of each row into chunks of length chunk_length to make use of batches more effectively.
            This makes tokenizing large files take less time as the entire content is not stored into one sample.
            Parts of the code are inspired from: https://huggingface.co/docs/datasets/process#split-long-examples

            Returns: {"chunks" : chunks} 
            where chunks is a list of sentences split after chunk_length characters.
        """
        chunks = []
        CHUNK_LENGTH = self.cfg.untokenized.chunk_size if self.cfg.untokenized.get("chunk_size") else 100
        for sentence in examples[self.cfg.tokenization_args.data_column_name]:
            chunks += [sentence[i:i + CHUNK_LENGTH] for i in range(0, len(sentence), CHUNK_LENGTH)]
        return {"chunks": chunks}

    def get_untokenized_files(self) -> List:
        """
            Returns all readable files from a given directory which are going to be used for tokenization. 
            To do so, the field "dataset_dir" from the hydra config is evaluated. All files from the directory "untokenized"
            within dataset_dir are returned. 
            If the directory does not exist, it is created and an empty list is returned.
        """
        log.debug(f'Checking for files with name containing {self.cfg.name} under directory: {self.untokenized_dir}')
        return [x for x in glob(self.RAW_DATA_DIR) if not os.path.isdir(x)]

    def create_hf_dataset(self):
        files = self.get_untokenized_files()
        #https://huggingface.co/docs/datasets/create_dataset#from-local-files
        def gen_samples():
            for filename in files:
                with open(filename, 'r') as file:
                    yield {"text": file.read()}
        all_files = datasets.Dataset.from_generator(gen_samples)
        all_files = all_files.map(
            self.chunk_examples, 
            batched=True, 
            batch_size = self.batch_size,
            remove_columns=["text"])
        all_files.rename_columns("chunks", "text") 
        dataset_dict = {}
        for split in DatasetSplitType:
            dataset_dict
        raise NotImplementedError()
        #return DatasetDict({"train": })

    def tokenize_data(self, untokenized_data: DatasetDict) -> None:
        #slice spits: https://huggingface.co/docs/datasets/loading#slice-splits
        #keys of DatasetDict are split types
        log.debug(f'Tokenizing data: {untokenized_data}')
        batch_size = self.cfg.untokenized.args.batches
        if batch_size is None:
            batch_size = 1000 #default for huggingface

        def tokenizer_function(batch):
            return {MODEL_INPUT_NAME: [tokenizer_singleton.tokenizer.encode(x) for x in batch[text_column_name]]}
        dump_file_names = {split.name: self.DUMP_FILE_PATH + self.CACHE_FILENAME_PREFIXES[split] + "tokenized.arrow" for split in DatasetSplitType}

        #tokenize them
        tokenized_datasets = untokenized_data.map(
            tokenizer_function,
            batched=True,
            #batch_size = batch_size,
            remove_columns=[text_column_name],
            desc="Running tokenizer on dataset",
            cache_file_names=dump_file_names
        )

    def get_untokenized_data(self, splits: List[DatasetSplitType]) -> DatasetDict:
        raise NotImplementedError()

    def get_collator(self) -> Callable:
        return None #dataloader collate_fn by default None
        


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