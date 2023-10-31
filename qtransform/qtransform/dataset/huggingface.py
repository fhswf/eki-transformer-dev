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
        dataset = load_dataset(cfg.name)
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
            return tokenizer(example["text"], max_length=1024, truncation=True)

        dataset = dataset.map(tokenization, batched=True)
        return dataset
    