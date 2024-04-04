import torch
import numpy as np
from torch.utils.data import Dataset
from qtransform.utils.introspection import get_classes, concat_paths, get_dtype
from qtransform.dataset import DatasetSplits, TokenizedDatasetGenerator, DatasetSplitType, MODEL_INPUT_NAME, MODEL_LABEL_NAME, MODEL_MASK_NAME
from qtransform.tokenizer import get_tokenizer, Tokenizer
import os
from logging import getLogger
from dataclasses import fields
from typing import Any, Union, Dict, Callable, Tuple, List
from omegaconf import DictConfig, open_dict
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes, concat_paths, get_dtype
import qtransform
from os import listdir, makedirs
from os.path import join, exists
from qtransform.tokenizer import Tokenizer, get_tokenizer
import datasets
from datasets.dataset_dict import DatasetDict
from qtransform.tokenizer.tokenizer_singleton import tokenizer_singleton
from qtransform import device_singleton
#TokenizedDatasetWrapper
import numpy as np
import torch
from pprint import PrettyPrinter

log = getLogger(__name__)

#TODO: it is possible to derive this class from HuggingfaceTokenizedDatasetGenerator
class FilesTokenizedDatasetGenerator(TokenizedDatasetGenerator):
    """
    TokenizedDatasetGenerator used to load files and tokenize them into numpy arrays.
    """
    DATASET_FILE_SUFFIX: str
    DATASET_FILE_PATH: str #by default: cache_dir from config

    def __init__(self, cfg: DictConfig):
        super().__init__(cfg)
        self.cfg = cfg
        self.DATASET_FILE_SUFFIX = "tokenized.bin"
        self.RAW_DATA_DIR = self.cfg.untokenized.args.cache_dir.replace('~', os.path.expanduser('~'))
        #make __post_init__?
        makedirs(self.DATASET_FILE_PATH, exist_ok=True)
        self.batch_size = self.cfg.untokenized.args.batches
        if self.batch_size is None:
            self.batch_size = 1000 #default for huggingface

    def get_tokenized_dataset(self) -> DatasetSplits:
        dataset_splits = DatasetSplits()
        for split in DatasetSplitType:
            try:
                dataset_splits[split] = MemmapDataset(self.get_filepath_split(split), block_size=self.block_size)
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
        for sentence in examples["text"]:
            chunks += [sentence[i:i + CHUNK_LENGTH] for i in range(0, len(sentence), CHUNK_LENGTH)]
        return {"chunks": chunks}

    def get_untokenized_files(self) -> List:
        """
            Returns all readable files from a given directory which are going to be used for tokenization. 
            To do so, the field "dataset_dir" from the hydra config is evaluated. All files from the directory "untokenized"
            within dataset_dir are returned. 
            If the directory does not exist, it is created and an empty list is returned.
        """
        log.debug(f'Retrieving files from: {self.RAW_DATA_DIR}')
        paths = []
        for x in os.listdir(self.RAW_DATA_DIR):
            path = os.path.join(self.RAW_DATA_DIR, x)
            if not os.path.isdir(path):
                paths.append(path)
        return paths

    def get_untokenized_data(self, splits: List[DatasetSplitType]) -> DatasetDict:
        assert isinstance(splits, list), f'splits {splits} is not a list'
        #assert list(filter(lambda x: isinstance(x, DatasetSplitType), splits))
        files = self.get_untokenized_files()
        log.debug(f'{files}')
        #https://huggingface.co/docs/datasets/create_dataset#from-local-files
        def gen_samples():
            for filename in files:
                with open(filename, 'r') as file:
                    yield {"text": file.read()}
        all_files: datasets.Dataset = datasets.Dataset.from_generator(gen_samples)
        all_files = all_files.map(
            self.chunk_examples, 
            batched=True, 
            batch_size = self.batch_size,
            remove_columns=["text"])
        all_files = all_files.rename_columns({"chunks": "text"})
        dataset_dict = {}
        for split in splits:
            mapping = split.name if split.name == "train" else "test"
            size = self.cfg.untokenized.splits[split.name.lower()]["size"]
            assert isinstance(size, float), f'Size for split {split.name} invalid ({size})'
            dataset_dict[split.name] = all_files.train_test_split(size)[mapping]
        log.debug(f'dataset_dict: {dataset_dict}')
        return DatasetDict(dataset_dict)

    def tokenize_data(self, untokenized_data: DatasetDict) -> None:
        #slice spits: https://huggingface.co/docs/datasets/loading#slice-splits
        #keys of DatasetDict are split types
        log.debug(f'Tokenizing data: {untokenized_data}')
        batch_size = self.cfg.untokenized.args.batches
        if batch_size is None:
            batch_size = 1000 #default for huggingface
        text_column_name = untokenized_data.column_names[list(untokenized_data.keys())[0]][0]
        log.debug(f'Using column name: {text_column_name}')

        def tokenizer_function(batch):
            batch_ids = [tokenizer_singleton.tokenizer.encode(x) for x in batch[text_column_name]]
            return {"input_ids": batch_ids, "length": [len(x) for x in batch_ids]}

        #tokenize them
        tokenized_datasets = untokenized_data.map(
            tokenizer_function,
            batched=True,
            #batch_size = batch_size,
            remove_columns=[text_column_name],
            desc="Running tokenizer on dataset"
        )
        #each sample of split contains amount of tokens to derive size of memmap
        length_splits = {split:sum(tokenized_datasets[split]["length"]) for split in tokenized_datasets}
        cache_file_names = {split.name: self.get_filepath_split(split) for split in DatasetSplitType}

        #MEMMAP processing begins here
        for split, data in tokenized_datasets.items():
            offset = 0
            try:
                log.debug(f'Split "{split}" has {length_splits[split]} tokens.')
                log.debug(f'Begin writing into memmap')
                path = cache_file_names[split]
                memmap = np.memmap(path, mode='w+', dtype=np.int64, shape=(length_splits[split], ))
                for batch_id in range(batch_size):
                    batch = tokenized_datasets[split].shard(num_shards=batch_size, index=batch_id)
                    if batch_id % 100 == 0:
                        log.debug(f'Batch: {batch_id}/{batch_size}. Length of batch: {len(batch)}')
                    if len(batch) == 0:
                        break
                    tokens = np.concatenate(batch["input_ids"], dtype=np.int64)
                    if hasattr(log, "trace"): log.trace(f'Writing into memmap from {offset}:{offset+len(tokens)}. Length of tokens: {len(tokens)}')
                    memmap[offset:offset+len(tokens)] = tokens
                    offset += len(tokens)
                memmap.flush()
            except Exception as e:
                #remove broken memmap file
                log.error(f'Stopped tokenization of split {split}.\nRemoving the broken memmap file under {path}', exc_info=True)
                os.remove(path)
                raise FileNotFoundError() #cannot continue running script as tokenized file has been removed
            log.debug(f'Tokenization done.')

    def get_intermediate_tokenized_data(self) -> Dict[DatasetSplitType,  Dict[str, Union[bool, str]]]:
        log.warning(f'Not implemented yet')
        raise None

    def get_collator(self) -> Callable:
        return None #dataloader collate_fn by default None, padding not necessary
        


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
        dtype = np.int64 #more than enough tokens (2^64 -1)
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

        return DatasetDict({MODEL_INPUT_NAME: inputs, MODEL_LABEL_NAME: labels, MODEL_MASK_NAME: torch.ones(self.block_size)})