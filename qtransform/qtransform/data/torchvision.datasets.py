
from torchvision import datasets, transforms
import torch

transform = transforms.Compose([ transforms.ToTensor(), transforms.Normalize((0.1307,),(0.3081,)) ])
tain      = datasets.MNIST('../data', train=True, download=True,transform=transform)
test      = datasets.MNIST('../data', train=False,transform=transform)

def load_mnist():
    def rf(train_kwargs, test_kwargs):
        train_loader = torch.utils.data.DataLoader(tain, **train_kwargs)
        test_loader  = torch.utils.data.DataLoader(test, **test_kwargs)
        return train_loader, test_loader
    return rf