from dataclasses import dataclass
from typing import Callable, List, Tuple
from qtransform.dataset import DatasetInfo, DatasetWrapper
from qtransform.utils.introspection import get_classes
import torch
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import numpy as np
import tiktoken
from dataclasses import fields
from omegaconf import DictConfig
from datasets import load_dataset  # huggingface datasets
from datasets import Dataset as HuggingfaceDataset # avoid naming conflict with Torch dataset

import logging
log = logging.getLogger(__name__)

class HuggingfaceDatasetWrapper(DatasetWrapper):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        HuggingfaceDataset.num_proc = os.cpu_count()/2
        #only load splits which are not empty (size above 0.0)
        for split in [x.name for x in fields(self.dataset_sizes)]:
            split_size = getattr(self.dataset_sizes, split)
            if split_size > 0.0:
                pass

    #TODO: REFACTOR
    def load_dataset(self) -> DatasetInfo:
        self.check_split(split)
        #https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes
        dataset_splits = load_dataset(self.cfg.name)
        #dataset = load_dataset("openwebtext") # takes 54GB in huggingface .cache dir, about 8M documents (8,013,769)
        from transformers import AutoTokenizer, GPT2TokenizerFast
        #tokenizer = AutoTokenizer.from_pretrained("gpt2",kwargs={"max_length": 1024})
        # TODO cfg this
        tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
        """
            t5: T5Tokenizer (T5 model)
            distilbert: DistilBertTokenizer (DistilBert model)
            albert: AlbertTokenizer (ALBERT model)
            camembert: CamembertTokenizer (CamemBERT model)
            xlm-roberta: XLMRobertaTokenizer (XLM-RoBERTa model)
            longformer: LongformerTokenizer (AllenAI Longformer model)
            roberta: RobertaTokenizer (RoBERTa model)
            bert-base-japanese: BertJapaneseTokenizer (Bert model)
            bert: BertTokenizer (Bert model)
            openai-gpt: OpenAIGPTTokenizer (OpenAI GPT model)
            gpt2: GPT2Tokenizer (OpenAI GPT-2 model)
            transfo-xl: TransfoXLTokenizer (Transformer-XL model)
            xlnet: XLNetTokenizer (XLNet model)
            xlm: XLMTokenizer (XLM model)
            ctrl: CTRLTokenizer (Salesforce CTRL model)
            electra: ElectraTokenizer (Google ELECTRA model)
        """
        def tokenization(example):
            # TODO cfg this
            #max_length is the length of the attention mask
            return tokenizer(example["text"], max_length=1024, truncation=True)
        #of type DatasetDict, contains all splits (test, validation, train)
        dataset_splits = dataset_splits.map(tokenization, batched=True)
        dataset_splits = dataset_splits.map()
        dataset_info = DatasetInfo(self.cfg.name)
        #TODO:  code is pretty similiar to FileSystemLLMDatasetWrapper, maybe modularise it
        #TODO:  some datasets have one row, others have multiple -> ''.join()

        #since data splits are split already, no need to worry about overlapping
        match split:
            #10.000 tokens, size is 0.7 -> start can be between 0.0 and 0.3
            case 'train':
                start_size = torch.randint(100 - round(self.self.dataset_sizes.train * 100), (1,)) / 100
                dataset_info.train = HuggingfaceDataset(dataset_splits["train"], self.cfg.args.block_size, start = start_size, end=self.dataset_sizes.train + start_size)
                if self.dataset_sizes.eval > 0.0:
                    eval_size = torch.randint(100 - round(self.self.dataset_sizes.eval * 100), (1,)) / 100
                    dataset_info.eval = HuggingfaceDataset(dataset_splits["validation"], self.cfg.args.block_size, start=eval_size, end=eval_size + self.dataset_sizes.eval)
            case 'bench':
                bench_size = torch.randint(100 - round(self.self.dataset_sizes.bench * 100), (1,)) / 100
                dataset_info.test = HuggingfaceDataset(dataset_splits["train"], self.cfg.args.block_size, start=bench_size, end=self.dataset_sizes.train + bench_size)
            case 'test':
                test_size = torch.randint(100 - round(self.self.dataset_sizes.bench * 100), (1,)) / 100
                dataset_info.test = HuggingfaceDataset(dataset_splits["test"], self.cfg.args.block_size, start= 1.0 - self.dataset_sizes.test)
        return dataset_info

    def shuffle(self):
        raise NotImplementedError()

class HuggingfaceDataset(Dataset):
    
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
