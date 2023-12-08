from dataclasses import dataclass
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

class TorchvisionDataset(DatasetInfo, DatasetWrapper):
    def __init__(self) -> None:
        pass
    def load_dataset(cfg: DictConfig) -> Tuple[Dataset, Dataset]:
        available_datasets = get_classes(datasets, Dataset)
        if cfg.name not in available_datasets:
            log.error(f"Dataset {cfg.name} not found in {datasets.__package__}")
            raise KeyError
        
        # TODO find good structure for all our data
        root_path = os.path.join(cfg.root_path, "data", "torchvision", "datasets", cfg.name)
        transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)) ])
        train = available_datasets[cfg.name](root=root_path, train=True, download=True, transform=transform)
        test = available_datasets[cfg.name](root=root_path, train=False, transform=transform)
        return train, test

    def get_dataloader() -> DataLoader:
        return None
