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
from datasets import load_dataset
from datasets.dataset_dict import DatasetDict

import logging
log = logging.getLogger(__name__)
#https://huggingface.co/docs/datasets/v2.15.0/en/package_reference/main_classes
class HuggingfaceDatasetWrapper(DatasetWrapper):
    """
        Retrieves a huggingface datasetand returns a DatasetInfo object. Under the hood, the datasets are tokenized and written
        into a numpy memmap file on the local user's harddrive for performance reasons. It also avoids having to load and tokenize 
        the same datasets multiple times.
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)

    def create_hf_dataset(self) -> DatasetDict:
        return load_dataset(self.cfg.name)

    def shuffle(self):
        raise NotImplementedError()