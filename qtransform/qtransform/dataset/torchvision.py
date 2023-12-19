from dataclasses import dataclass, fields
from typing import Callable, Tuple
from qtransform.dataset import DatasetInfo, DatasetWrapper
from qtransform.utils.introspection import get_classes
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
import os
from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)

class TorchvisionDatasetWrapper(DatasetWrapper):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg)
        self.available_datasets = get_classes(datasets, Dataset)
        if self.cfg.name not in self.available_datasets:
            log.error(f"Dataset {self.cfg.name} not found in {datasets.__package__}")
            raise KeyError

    def load_dataset(self):
        # TODO find good structure for all our data
        root_path = os.path.join(self.cfg.root_path, "torchvision", self.cfg.name)
        transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)) ])
        #TODO: implement eval
        self.dataset_info.train = self.available_datasets[self.cfg.name](root=root_path, train=True, download=True, transform=transform)
        self.dataset_info.test = self.available_datasets[self.cfg.name](root=root_path, train=False, transform=transform)
    
    def shuffle(self):
        raise NotImplementedError()