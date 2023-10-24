from dataclasses import dataclass
from typing import Callable
from qtransform.dataset import DatasetInfo, DatasetWrapper
from qtransform.utils.introspection import get_classes
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from omegaconf import DictConfig
from datasets import load_dataset  # huggingface datasets

import logging
log = logging.getLogger(__name__)

class HuggingfaceDataset(DatasetInfo, DatasetWrapper):
    def __init__(self) -> None:
        HuggingfaceDataset.num_proc = os.cpu_count()/2
        pass

    @classmethod
    def load_dataset(cls, cfg: DictConfig) -> Dataset:
        #dataset = load_dataset("openwebtext") # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
        dataset = load_dataset(cfg.name)
        dataset = dataset.with_format('torch')
        data_split = dataset["train"].train_test_split(test_size=0.0005, seed=2345, shuffle=True)
        split['val'] = split.pop('test')  

        # tokenize the dataset:
        enc = tiktoken.get_encoding("gpt2")
        def encode(example):
            ids = enc.encode_ordinary(example['text']) 
            ids.append(enc.eot_token) #eot token 50256 for gpt2 bpe
            return {'ids': ids, 'len': len(ids)}
        tokenized = data_split.map(
            encode,
            remove_columns=['text'],
            desc="tokenizing the splits",
            num_proc=cls.num_proc,
        )
        # TODO make this a stream of some sorts
        # concatenate all the ids in each dataset into one large file we can use for training
        for split, dset in tokenized.items():
            arr_len = np.sum(dset['len'])
            filename = os.path.join(os.path.dirname(__file__), f'{split}.bin')
            dtype = np.uint16 # (can do since enc.max_token_value == 50256 is < 2**16)
            arr = np.memmap(filename, dtype=dtype, mode='w+', shape=(arr_len,))
            total_batches = 1024
            idx = 0
            for batch_idx in tqdm(range(total_batches), desc=f'writing {filename}'):
                # Batch together samples for faster write
                batch = dset.shard(num_shards=total_batches, index=batch_idx, contiguous=True).with_format('numpy')
                arr_batch = np.concatenate(batch['ids'])
                # Write into mmap
                arr[idx : idx + len(arr_batch)] = arr_batch
                idx += len(arr_batch)
            arr.flush()
        return None
     
    def get_loader(self, cfg: DictConfig):
        def get_batch(split):
            data = train_data if split == 'train' else val_data
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
            if device_type == 'cuda':
                # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
                x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
            else:
                x, y = x.to(device), y.to(device)
            return x, y
        return get_batch

