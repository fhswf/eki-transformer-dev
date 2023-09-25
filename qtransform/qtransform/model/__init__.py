
from omegaconf import DictConfig
from torch import nn

def get_model(cfg: DictConfig) -> nn.Module:
    """ get model info and return a configured torch nn.Module instance """

    return 