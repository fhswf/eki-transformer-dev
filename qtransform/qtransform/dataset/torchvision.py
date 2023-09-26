from typing import Callable
from qtransform.utils.introspection import get_classes
from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader, Dataset
import os
from omegaconf import DictConfig
import logging
log = logging.getLogger(__name__)

def load_dataset(name, cfg: DictConfig) -> Dataset:
    available_datasets = get_classes(datasets, Dataset)
    if name not in available_datasets:
        log.error(f"Dataset {name} not found in {datasets.__package__}")
        raise KeyError
    
    # TODO find good structure for all our data
    root_path = os.path.join(cfg.root_path, "data", "torchvision", "datasets", name)
    transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)) ])
    train = available_datasets[name](root=root_path, train=True, download=True, transform=transform)
    test = available_datasets[name](root=root_path, train=False, transform=transform)
    return train, test
