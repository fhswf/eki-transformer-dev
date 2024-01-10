from typing import Any, Tuple, List
import torch
import numpy as np
from torch.utils.data import Dataset
from omegaconf import DictConfig, open_dict
from qtransform.utils.introspection import get_classes, concat_paths
from qtransform.dataset import DatasetInfo, DatasetWrapper
from qtransform.dataset.tokenizer import get_tokenizer, Tokenizer
import os
from glob import glob
import logging
from dataclasses import fields
import datasets
from datasets.dataset_dict import DatasetDict

log = logging.getLogger(__name__)

class FileSystemLLMDatasetWrapper(DatasetWrapper):
    """
        DatasetWrapper used to load .bin files from a dataset file and return a DatasetInfo object containing torch.utils.Dataset instances.
        They can be iterated over with a Dataloader, making the process of retrieving data abstract.
    """
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        #directories for untokenized and tokenized files
        self.untokenized_dir = concat_paths([*cfg.dataset_dir, "untokenized", ""])

    def create_hf_dataset(self) -> DatasetDict:
        #choose name "text" as feature name
        with open_dict(self.cfg):
            self.cfg.args["data_column_name"] = "text"
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

    def shuffle(self):
        raise NotImplementedError()
