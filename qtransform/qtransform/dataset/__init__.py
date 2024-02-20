from typing import Any, Union, Dict
from omegaconf import DictConfig, open_dict
import logging
from torch.utils.data import Dataset, DataLoader
from qtransform.utils.introspection import _get_module, get_classes, concat_paths, get_dtype
import qtransform
from qtransform import classloader
from dataclasses import dataclass, fields
from enum import Enum
from os.path import join
from qtransform.dataset.tokenizer import Tokenizer, get_tokenizer
import datasets
from datasets.dataset_dict import DatasetDict

log = logging.getLogger(__name__)

class DatasetRunType(Enum):
    TRAIN = "train"
    EVAL = "eval"
    BENCH = "bench"

@dataclass
class DatasetSizes:
    """
        Remember the size of datasets in percent to retain the size when shuffling and to 
        do checks before data should be loaded. 
    """
    train: float = 0.0 #size of training data
    eval: float = 0.0 #size of the subset of training data to check if model is overfitting
    bench: float = 0.0 

    def __post_init__(self):
        empty_split = 0 #check how many splits are not empty
        count = 0
        for field in fields(self):
            attr = getattr(self, field.name)
            if not isinstance(attr, float):
                log.error(f'{field.name} is not a floating point number.')
                raise KeyError()
            if attr < 0.0 or attr > 1.0:
                log.error(f'Data split {field.name} has to be within range 0.0 and 1.0, not {attr}')
                raise ValueError()
            if attr == 0.0:
                empty_split += 1
            count += 1
        if self.eval > 0.0 and self.train == 0.0:
            log.error(f'Cannot validate training if size of training data is empty')
            raise ValueError()
        elif self.eval == 1.0:
            log.warning(f'Size of validation split is equal to training split (eval = 1.0)')
        #all fields are 0.0, no point in continuing process as no data can be loaded
        if count == empty_split:
            log.error(f'Sizes of specified splits are zero.')
            raise ValueError()

@dataclass
class DatasetInfo:
    """
        Dataclass containing the datasets for training, eval, testing, benchmark along with the name of the dataset.
        After construction, a simple type check is done with the __post_init__ hook.
    """
    name: str
    train: Dataset = None
    eval: Dataset = None
    test: Dataset = None
    bench: Dataset = None

    """def __setattr__(self, __name, __value):
        if __name not in self.fields.keys():
            log.error(f'DatasetInfo should only contain fields: {self.fields.keys()}')
            raise KeyError()
        field = self.fields[__name]
        current_attr = getattr(self, field.name)
        if current_attr is not None and not isinstance(current_attr, field.type):
                log.error(f'DatasetInfo field {field.name} expects field type {field.type}, not {type(current_attr)}')
                raise TypeError()"""

    def __post_init__(self):
        self.fields = {x.name:x for x in fields(self)}
        for field in self.fields.values():
            current_attr = getattr(self, field.name)
            if current_attr is not None and not isinstance(current_attr, field.type):
                log.error(f'DatasetInfo field {field.name} expects field type {field.type}, not {type(current_attr)}')
                raise TypeError()

from abc import ABC, abstractmethod
import os

class DatasetWrapper(ABC):
    """
    Capsule around Dataset, to unify their interfaces.
    Each DatasetWrapper has to contain a method to (down)load the data, create a Dataloader, 
    and provide information on whether the dataset contained in this wrapper provides training, eval/test or benchmark data.
    """

    dataset_sizes: DatasetSizes = None
    cfg: DictConfig = None

    def __init__(self, cfg: DictConfig):
        self.cfg: DictConfig = cfg
        if self.cfg.get('name') is None:
            log.error(f'No dataset name specified.')
            raise KeyError()
        if self.cfg.get('sizes') is None:
            log.warning(f'No sizes for the data splits specified.')
            raise KeyError()
        #add empty args property to avoid None checking every time
        if self.cfg.get('args') is None:
            with open_dict(self.cfg):
                self.cfg["args"] = {}
        self.dataset_sizes = DatasetSizes(**cfg.sizes)
        self.dataset_info = DatasetInfo(name=self.cfg.name)
        self.tokenized_dir = concat_paths([*cfg.dataset_dir, "tokenized", cfg.tokenizer.encoding])
        self.dataset_file = join(self.tokenized_dir, self.cfg.name+ '-' + self.cfg.tokenizer.dtype + '.bin')
        #currently, dtype has to be set by user. maybe it could also be automatically infered by the max tokens property of Tokenizer
        if self.cfg.tokenizer.get('dtype') is None:
            log.debug(f'Dtype for dataset omited. Assuming default: int64')
            self.dtype = get_dtype('int64')
        else:
            self.dtype = get_dtype(self.cfg.tokenizer.dtype)
        #tokenizer is created regardless of whether tokenization is necessary or not
        #reason: typechecking of metadata to save it in model checkpoints
        self.tokenizer: Tokenizer = get_tokenizer(self.cfg.tokenizer)

    def load_dataset(self) -> DatasetInfo:
        """
            Loads a dataset from the config specified in the constructor. The split argument specifies
            the size of the returned dataset which have been stored in an instance of type DataSizes. 
            If a dataset has not been tokenized yet, it will be tokenized and saved under the directory specified in 
            dataset_dir of the hydra config. The tokenization uses huggingface's mapping functionality.
            The tokenization was inspired by Karpathy's openwebtext script in nanoGPT.
            (https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py#L80)
        """
        log.info(f'Loading dataset: {self.cfg.name}, with encoding: {self.cfg.tokenizer.encoding} and dtype: {self.dtype}')
        if not os.path.exists(self.dataset_file):
            #assume feature "text" contains text to be tokenized
            if self.cfg.args.get('data_column_name') is None:
                log.warning(f'Field data_column_name omited. Assuming feature name "text" to be containing training data.')
                with open_dict(self.cfg):
                    self.cfg.args["data_column_name"] = "text"

            log.info(f'Begin tokenizing dataset as file: {self.dataset_file}')
            os.makedirs(self.tokenized_dir, exist_ok = True)
            #retrieve dataset 
            dataset_splits: DatasetDict = self.create_hf_dataset()
            log.debug(f'Begin tokenizing dataset.')

            #default huggingface batch_size: 1000 (https://huggingface.co/docs/datasets/process#batch-processing)
            batch_size = self.cfg.args.get('batches')
            if batch_size is None or batch_size > len(dataset_splits):
                batch_size = 1000 
                log.info(f'Using batch size {batch_size} for memmap processing')

            #concatenate splits into one large split, only use text column
            #otherwise, chunking will fail due to uneven amount of samples in each feature (https://github.com/huggingface/datasets/issues/1817#issuecomment-774066254)
            dataset_splits = datasets.concatenate_datasets([x.select_columns(self.cfg.args.data_column_name) for x in dataset_splits.values()])
            log.debug(f'Dataset has {len(dataset_splits)} rows.')
            tokenizer = get_tokenizer(self.cfg.tokenizer)
            #split samples into sentences of length chunk_size or simply rename feature to "chunk" if chunking is False
            chunking = self.cfg.args.get("chunking", False)
            if chunking is True:
                log.debug(f'Begin chunking dataset into sentences of length {self.cfg.args.chunk_size}')
                dataset_splits = dataset_splits.map(
                    self.chunk_examples, 
                    batched=True, 
                    batch_size = batch_size,
                    num_procs = self.get_hf_num_proc(dataset_splits.num_rows),
                    remove_columns = self.cfg.args.data_column_name) 
            else:
                #saves if-else statements for feature name
                log.info(f'Skipping chunking of datasets')
                dataset_splits = dataset_splits.rename_column(self.cfg.args.data_column_name, "chunks")
            log.debug(f'Dataset after chunking: {dataset_splits}')

            #larger batch size than amount of samples will lead to one large batch containing all samples
            if len(dataset_splits) < batch_size:
                log.warning(f'Batch size of {batch_size} and sample size of {len(dataset_splits)} makes batch processing redundant.')
            def encode_batch(batch):
                batch_ids = [tokenizer.encode(x) for x in batch["chunks"]]
                return {"input_ids": batch_ids, "length": [len(x) for x in batch_ids]}

            dataset_splits = dataset_splits.map(
                #map function expects dictionary or dataset object, tokenize function returns list of tokens (integers)
                encode_batch,
                batched=True,
                batch_size = batch_size, 
                remove_columns = "chunks",
                num_proc=self.get_hf_num_proc(dataset_splits.num_rows), 
                desc=f'tokenizing from chunks')
            if hasattr(log, "trace"): log.trace(f'Dataset split after tokenization: {dataset_splits}')
            length_tokens = sum(dataset_splits["length"]) 
            tokenizer.meta.num_tokens = length_tokens
            log.debug(f'Dataset has {length_tokens} tokens.')
            log.debug(f'Begin writing into memmap')

            #write tokens into memmap
            offset = 0
            try:
                memmap = np.memmap(self.dataset_file, mode='w+', dtype=self.dtype, shape=(length_tokens, ))
                for batch_id in range(batch_size):
                    batch = dataset_splits.shard(num_shards=batch_size, index=batch_id)
                    log.debug(f'Batch: {batch_id}/{batch_size}. Length of batch: {len(batch)}')
                    if len(batch) == 0:
                        break
                    #would write operation be faster if values are moved to gpu?
                    tokens = np.concatenate(batch["input_ids"], dtype=self.dtype)
                    if hasattr(log, "trace"): log.trace(f'Writing into memmap from {offset}:{offset+len(tokens)}. Length of tokens: {len(tokens)}')
                    memmap[offset:offset+len(tokens)] = tokens
                    offset += len(tokens)
                memmap.flush()
                tokenizer.save_metadata(self.tokenized_dir)
            except Exception as e:
                #remove broken memmap file
                log.error(f'Stopped tokenization due to error: {error}.\nRemoving the broken memmap file under {self.dataset_file}')
                os.remove(self.dataset_file)
                raise FileNotFoundError() #cannot continue running script as tokenized file has been removed
            log.debug(f'Tokenization done.')

        if self.dataset_sizes.train > 0.0:
            self.dataset_info.train = MemmapDataset(self.dataset_file, self.dtype, self.cfg.args.block_size, end=self.dataset_sizes.train)
        #eval
        if self.dataset_sizes.eval > 0.0:
            percentage_eval = round(self.dataset_sizes.eval * 100)
            #eval_start = torch.randint(round(self.dataset_sizes.train * 100) - percentage_eval, (1, )).item() / 100
            eval_start = self.dataset_sizes.train
            if not isinstance(eval_start, float) or eval_start <= 0.0:
                log.error(f'Cannot eval without training data (train size specified was: {eval_start})')
                raise ValueError()
            eval_end = min(1.0, eval_start + self.dataset_sizes.eval)
            #self.dataset_info.eval = MemmapDataset(self.dataset_file, self.dtype, self.cfg.args.block_size, start=eval_start, end=eval_start + self.dataset_sizes.eval)
            self.dataset_info.eval = MemmapDataset(self.dataset_file, self.dtype, self.cfg.args.block_size, start=eval_start, end=eval_end)
        #bench
        if self.dataset_sizes.bench > 0.0:
            self.dataset_info.bench = MemmapDataset(self.dataset_file, self.dtype, self.cfg.args.block_size, end=self.dataset_sizes.bench)
        
    def get_hf_num_proc(self, rows: int) -> int:
        """
            Get num_proc argument for huggingface mapping function based on the amount of rows that a dataset has.
            It is at maximum os.cpu_count // 2 and at the minimum rows.
        """
        #TODO: vocab not saved with multithreading currently
        max_procs = os.cpu_count() // 2
        if self.cfg.tokenizer.encoding == "character":
            return 1
        elif max_procs <= rows:
            return max_procs
        return rows

    @classmethod
    @abstractmethod
    def create_hf_dataset(self) -> DatasetDict:
        """
            Creates a huggingface dataset from text files or loads a pre-existing huggingface dataset. Huggingface datasets
            are used due to the memory efficient implementation as well as their mapping functionality.
        """
        pass

    def chunk_examples(self, examples):
        """
            Splits the text of each row into chunks of length chunk_length. 
            It is useful when samples have large amounts of text in order to perform
            mapping in batches more efficiently.
            Parts of the code are inspired from: https://huggingface.co/docs/datasets/process#split-long-examples

            Returns: {"chunks" : chunks} 
            where chunks is a list of sentences split after chunk_length characters.
        """
        chunks = []
        CHUNK_LENGTH = self.cfg.args.chunk_size if self.cfg.args.get("chunk_size") else 100
        for sentence in examples[self.cfg.args.data_column_name]:
            chunks += [sentence[i:i + CHUNK_LENGTH] for i in range(0, len(sentence), CHUNK_LENGTH)]
        return {"chunks": chunks}
        

    def check_split(self, split: str):
        """
            Checks whether the DatasetSizes dataclass contains a field with name split.
        """
        splits = [x.name for x in fields(DatasetSizes)]
        if split not in splits:
            log.error(f'Datasets can only be split among {splits}, not {split}')
            raise ValueError()

    @classmethod
    @abstractmethod
    def shuffle(self):
        """
            Idea:   Have an instance of type dataset (named all_datasets), its size is the sum of all instantiated datasets in DatasetInfo
                    E.g. training, eval dataset each are 10MB, all_datasets is 20MB of size
                    When shuffle is called, the training and eval datasets are created from all_datasets, containing different tensors than before.
            TODO:   Test the behavior of non-sequential datasets (test dataset goes from 90-100% and from 0-10%)
            TODO:   check size of dataset and splits, pick random number with torch.rand
        """
        pass

import numpy as np
from typing import Tuple
import torch

class MemmapDataset(Dataset):
    
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

def get_data(dataset_cfg: DictConfig) -> DatasetWrapper:
    import qtransform.dataset as package_self
    dataset_wrapper: DatasetWrapper = classloader.get_data(log, package_self, dataset_cfg.wrapper, DatasetWrapper, args={"cfg": dataset_cfg})
    return dataset_wrapper

def get_loader(dataloader_cfg: DictConfig, data:Dataset) -> DataLoader:
    log.debug(f"get_loader config: {dataloader_cfg}")
    loader = DataLoader(data, **dataloader_cfg)
    return loader