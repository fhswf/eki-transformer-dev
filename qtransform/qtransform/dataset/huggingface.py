from dataclasses import dataclass
from typing import Callable, List, Tuple
from qtransform.dataset import DatasetInfo, DatasetWrapper
from qtransform.dataset.tokenizer import get_tokenizer
from qtransform.utils.introspection import get_classes
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
from omegaconf import DictConfig, open_dict
from datasets import load_dataset, concatenate_datasets
from datasets.dataset_dict import DatasetDict
from datasets import Dataset as HuggingfaceDataset # avoid naming conflict with Torch datasets
from qtransform.dataset.files import _FileSystemLLMDataset

import logging
log = logging.getLogger(__name__)
#https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes
class HuggingfaceDatasetWrapper(DatasetWrapper):
    """
        Retrieves a huggingface datasetand returns a DatasetInfo object. Under the hood, the datasets are tokenized and written
        into a numpy memmap file on the local user's harddrive for performance reasons. It also avoids having to load and tokenize 
        the same datasets multiple times.
        The implementation was inspired by Karpathy's nanoGPT preparation script for tokenizing openwebtext
        (https://github.com/karpathy/nanoGPT/blob/master/data/openwebtext/prepare.py#L80)
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        HuggingfaceDataset.num_proc = os.cpu_count()/2
        if self.cfg.args.get('data_column_name') is None:
            log.warning(f'Field data_column_name omited. Assuming column "text" to contain training data.')
            with open_dict(self.cfg):
                self.cfg.args["data_column_name"] = "text"

    #TODO: each dataset has a json file containing metadata information. this could be useful within the meta.pkl file
    #https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/loading_methods#datasets.load_dataset
    def load_dataset(self) -> DatasetInfo:
        log.info(f'Loading dataset: {self.cfg.name}, with encoding: {self.cfg.tokenizer.encoding} and dtype: {self.dtype}')
        if not os.path.exists(self.dataset_file):
            log.info(f'Begin tokenizing dataset as file: {self.dataset_file}')
            os.makedirs(self.tokenized_dir, exist_ok = True)
            data_column_name = self.cfg.args.data_column_name
            dataset_splits: DatasetDict = load_dataset(self.cfg.name)
            #some datasets (tiny_shakespeare) cram their dataset in one row which makes batch processing redundant
            for split_name in dataset_splits.keys():
                if len(dataset_splits[split_name]) == 1:
                    log.warning(f'Sample length of {split_name} makes batch processing redundant as data is crammed into one row')
            log.debug(f'Begin tokenizing dataset.')
            def chunk_examples(examples):
                #splits the text of each row into chunks of length chunk_length. currently it is only used
                #for character tokenization to avoid feeding large samples to the tokenizer
                chunk_length = 100
                #perform tokenization on a handful of characters at a time
                #from: https://huggingface.co/docs/datasets/process#split-long-examples
                chunks = []
                for sentence in examples[data_column_name]:
                    chunks += [sentence[i:i + chunk_length] for i in range(0, len(sentence), chunk_length)]
                return {"chunks": chunks}

            #concatenate splits into one large split, only use text column
            dataset_splits = concatenate_datasets([x.select_columns(self.cfg.args.data_column_name) for x in dataset_splits.values()])
            log.debug(f'Dataset has {len(dataset_splits)} rows.')
            tokenizer = get_tokenizer(self.cfg.tokenizer)
            if self.cfg.tokenizer.encoding == 'character':
                log.debug(f'Begin chunking dataset into sentences of length 100')
                #split individual rows into blocks of 100 characters
                dataset_splits = dataset_splits.map(chunk_examples, batched=True, remove_columns = data_column_name)
                log.debug(f'Dataset after chunking: {dataset_splits}')
                log.debug(f'Example of the first chunk: "{dataset_splits["chunks"][0]}"')
                dataset_splits = dataset_splits.map(
                    #map function expects dictionary or dataset object, tokenize function returns list of tokens (integers)
                    lambda batch: {"input_ids": [tokenizer.encode(x) for x in batch["chunks"]]}, 
                    batched=True, 
                    remove_columns = "chunks",
                    desc="tokenizing the dataset from chunks")
            else:
                dataset_splits = dataset_splits.map(
                    #map function expects dictionary or dataset object, tokenize function returns list of tokens (integers)
                    lambda batch: {"input_ids": [tokenizer.encode(x) for x in batch[data_column_name]]},
                    batched=True, 
                    remove_columns = data_column_name,
                    desc = "tokenizing the dataset")
            if hasattr(log, "trace"): log.trace(f'Dataset split after tokenization: {dataset_splits}')
            first_example = dataset_splits["input_ids"][0] #for logging purposes
            first_example = first_example if len(first_example) < 50 else first_example[:50]
            #log.debug(f'First example: {first_example}')
            #after concatenation, length is the total amount of tokens in entire dataset
            length_tokens = tokenizer.meta.num_tokens
            log.debug(f'Dataset has {length_tokens} tokens.')
            batch_size = self.cfg.args.get('batches')
            if batch_size is None or batch_size > len(dataset_splits):
                batch_size = max(len(dataset_splits), len(dataset_splits) // 10)
                log.info(f'Using batch size {batch_size} for memmap processing')
            log.debug(f'Begin writing into memmap')
            #write tokens into memmap
            memmap = np.memmap(self.dataset_file, mode='w+', dtype=self.dtype, shape=(length_tokens, ))
            offset = 0
            try:
                for batch_id in range(batch_size):
                    batch = dataset_splits.shard(num_shards=batch_size, index=batch_id)
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
                log.error(f'Something went wrong while tokenizing the dataset. Reason: {e}.\nRemoving the broken memmap file under {self.dataset_file}')
                os.remove(self.dataset_file)
                raise FileNotFoundError() #cannot continue running script as tokenized file has been removed
            log.debug(f'Tokenization done.')

        #for now, exactly the same as FileSystemLLMDatasetWrapper. TODO: refactor
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

    def shuffle(self):
        raise NotImplementedError()

#@DeprecationWarning()
class HuggingfaceDataset(Dataset):
    """
        Huggingface uses FileSystemLLMDataset as the tokenized files are stored as a binary file and can be loaded with numpy.memmap
    """
    def __init__(self, hf_dataset_raw: HuggingfaceDataset, block_size: int, start: float = 0.0, end: float = 0.0):
        """
            Dataset containing huggingface data which are usable for large language models. This class should not be 
            instantiated directly, instead use the HuggingfaceDatasetWrapper in order to specify the
            dataset name. 
        """
        super().__init__()
        if not isinstance(block_size, int) or block_size <= 0:
            log.error(f'block_size has to be a positive integer, not {block_size}')
            raise ValueError()
        if not isinstance(hf_dataset_raw, HuggingfaceDataset):
            log.error(f'Dataset cannot be instantiated with: {hf_dataset_raw}')
            raise TypeError()
        self.hf_dataset_raw = hf_dataset_raw
        if self.hf_dataset_raw.get('input_ids') is None:
            log.error(f'Hugging face dataset was not tokenized.')
            raise KeyError()
        self.data = torch.Tensor()
        if block_size > self.length:
            log.warning(f'block_size is equal to the dataset length')
        #TODO:  huggingface datasets contain splits train, validation and test
        #       inside of each split, a property "text" contains an array of untokenized sentences
        #       after tokenization, each token is a seperate tensor
        #       -> only consider input_ids, construct tensors of block_size length
        #       e.g. split contains text: [["hallo welt"], "das ist ein test"]
        #       then the tokenized list input_ids contains: [[Tensor(id_hallo), Tensor(id_welt)], [Tensor(id_das), ...]]
        #       -> truncate tensors of each word into one large tensor, seperate at block_size, make input_ids one large tensor containing blocks

    def __len__(self) -> int:
        return self.hf_dataset_raw.num_rows
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        if index < 0 or index > self.length -1:
            log.error(f'Cannot retrieve index {index}')
            raise ValueError()
