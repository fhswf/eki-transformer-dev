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

log = logging.getLogger(__name__)

class MemmapTokenizedDatasetGenerator(TokenizedDatasetGenerator):
    """
    Creates and loads one dimensional numpy arrays from local disk. The numpy arrays contain the input ids of a dataset and are
    returned as a DatasetSplit instance. 
    """
    def __init__(self, tokenized_cfg: DictConfig, untokenized_cfg: DictConfig, *args, **kwargs) -> None:
        super().__init__()
        self.dataset_files = {}
        #TODO: tokenizer
        dataset_name = self.cfg.subset if self.cfg.subset is not None else self.cfg.name
        #<split>-<dataset>-<dtype>.bin e.g.: eval-openwebtext-float32.bin
        for split in ["train", "eval", "bench"]:
            self.dataset_files[split] = join(self.tokenized_dir, split + "-" + dataset_name + '-' + self.cfg.tokenizer.dtype + '.bin')
        #currently, dtype has to be set by user. maybe it could also be automatically infered by the max tokens property of Tokenizer
        if self.cfg.tokenizer.get('dtype') is None:
            log.debug(f'Dtype for dataset omited. Assuming default: int64')
            self.dtype = get_dtype('int64')
        else:
            self.dtype = get_dtype(self.cfg.tokenizer.dtype)


    def get_tokenized_dataset(self, *args, **kwargs) -> DatasetSplits:
        path_train = ""
        path_eval = ""
        path_benchmark = ""
        dtype = np.int64 #up to 2^64 tokens possible in vocabulary. TODO: make reducing vocab size possible
        raise NotImplementedError
        return DatasetSplits(bench=MemmapDataset(path_benchmark), train=MemmapDataset(path_train), eval=MemmapDataset(path_eval))
    
    def prepare_data(self) -> None:
        #copy of untokenized splits to check if hf dataset has to be created
        #TODO: dataset_splits could be an attribute that is set if it was None instead of doing this
        untokenized_splits = {}
        for split, path in self.dataset_files.items():
            if not os.path.exists(path):
                untokenized_splits[split] = path
        #splits have been tokenized
        if len(untokenized_splits) == 0:
            return
        os.makedirs(self.tokenized_dir, exist_ok = True)
        #retrieve dataset 
        dataset_splits: DatasetDict = self.create_hf_dataset()
        if "train" not in dataset_splits.keys():
            log.error(f'Dataset split does not contain train split')
            raise RuntimeError()
        #rename splits 
        for split in ["train", "eval", "bench"]:
            cfg_split = self.untokenized_cfg.splits.names[split]
            if cfg_split is None or dataset_splits.get(cfg_split, None) is None:
                continue
            dataset_splits[split] = dataset_splits.pop(self.cfg.splits.names[split])
        log.info(f'Begin tokenizing dataset {self.cfg.name}')
        with open_dict(self.untokenized_cfg):
            self.cfg.tokenization_args["data_column_name"] = "text"
        log.debug(f'{dataset_splits}')
        #default huggingface batch_size: 1000 (https://huggingface.co/docs/datasets/process#batch-processing)
        batch_size = self.cfg.tokenization_args.get('batches')
        if batch_size is None or batch_size > len(dataset_splits):
            batch_size = 1000 
            log.info(f'Using batch size {batch_size} for memmap processing')
        
        #only use columns which contain data, ignore e.g. labels for classification problems
        dataset_splits = dataset_splits.select_columns(self.cfg.tokenization_args.data_column_name)
        log.debug(f'Dataset has {len(dataset_splits.keys())} splits, with the splits having {[len(dataset_splits[x]) for x in dataset_splits.keys()]} samples.')
        
        
        ####################################
        #TODO: tokenizer
        tokenizer = get_tokenizer(self.cfg.tokenizer)
        #TODO: tokenizer
        ####################################
        
        
        dataset_splits = dataset_splits.map(
            self.chunk_examples, 
            batched=True, 
            batch_size = batch_size,
            num_proc = self.get_hf_num_proc(sum(dataset_splits.num_rows.values())),
            remove_columns = self.cfg.tokenization_args.data_column_name,
            desc=f'Chunking dataset into samples of length {self.cfg.tokenization_args.chunk_size}') 
        log.debug(f'Dataset after chunking: {dataset_splits}')

        #check if some splits are missing
        #do this after chunking as some datasets have 1 large sample
        missing_splits = {"eval", "bench"} - set(dataset_splits.keys())
        for missing_split in missing_splits:
            log.debug(f'Dataset is missing split {missing_split}. Deriving split from train split')
            split_dataset = dataset_splits["train"].train_test_split(getattr(self.dataset_sizes, missing_split), seed=2357)

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
            num_proc = self.get_hf_num_proc(sum(dataset_splits.num_rows.values())),
            desc=f'tokenizing from chunks')
        if hasattr(log, "trace"): log.trace(f'Dataset split after tokenization: {dataset_splits}')
        #each sample of split contains amount of tokens to derive size of memmap
        length_splits = {split:sum(dataset_splits[split]["length"]) for split in dataset_splits}
        tokenizer.meta.num_tokens = sum(length_splits.values())
        
        #MEMMAP processing begins here
        for split, path in untokenized_splits.items():
            offset = 0
            try:
                log.debug(f'Split "{split}" has {length_splits[split]} tokens.')
                log.debug(f'Begin writing into memmap')
                memmap = np.memmap(path, mode='w+', dtype=self.dtype, shape=(length_splits[split], ))
                for batch_id in range(batch_size):
                    batch = dataset_splits[split].shard(num_shards=batch_size, index=batch_id)
                    log.debug(f'Batch: {batch_id}/{batch_size}. Length of batch: {len(batch)}')
                    if len(batch) == 0:
                        break
                    tokens = np.concatenate(batch["input_ids"], dtype=self.dtype)
                    if hasattr(log, "trace"): log.trace(f'Writing into memmap from {offset}:{offset+len(tokens)}. Length of tokens: {len(tokens)}')
                    memmap[offset:offset+len(tokens)] = tokens
                    offset += len(tokens)
                memmap.flush()
                tokenizer.save_metadata(self.tokenized_dir)
            except Exception as e:
                #remove broken memmap file
                log.error(f'Stopped tokenization of split {split}.\nRemoving the broken memmap file under {path}', exc_info=True)
                os.remove(path)
                raise FileNotFoundError() #cannot continue running script as tokenized file has been removed
            log.debug(f'Tokenization done.')

    def create_hf_dataset(self):
        #choose name "text" as feature name
        with open_dict(self.cfg):
            #ignore data_column_name from config as hf dataset has been created on the fly
            self.cfg.tokenization_args["data_column_name"] = "text"
        files = self.get_untokenized_files()
        #https://huggingface.co/docs/datasets/create_dataset#from-local-files
        def gen_samples():
            for filename in files:
                with open(filename, 'r') as file:
                    yield {"text": file.read()}
        return DatasetDict({"train": datasets.Dataset.from_generator(gen_samples)})

    def get_untokenized_files(self) -> List:
        """
            Returns all readable files from a given directory which are going to be used for tokenization. 
            To do so, the field "dataset_dir" from the hydra config is evaluated. All files from the directory "untokenized"
            within dataset_dir are returned. 
            If the directory does not exist, it is created and an empty list is returned.
        """
        if not os.path.exists(self.untokenized_dir):
            log.debug(f'Creating directory {self.untokenized_dir}')
            os.makedirs(self.untokenized_dir, exist_ok=True)
            return []
        log.debug(f'Checking for files with name containing {self.cfg.name} under directory: {self.untokenized_dir}')
        return [x for x in glob(self.untokenized_dir + self.cfg.name + '*') if not os.path.isdir(x)]


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
        CHUNK_LENGTH = self.cfg.tokenization_args.chunk_size if self.cfg.tokenization_args.get("chunk_size") else 100
        for sentence in examples[self.cfg.tokenization_args.data_column_name]:
            chunks += [sentence[i:i + CHUNK_LENGTH] for i in range(0, len(sentence), CHUNK_LENGTH)]
        return {"chunks": chunks}

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