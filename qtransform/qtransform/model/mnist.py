import torch
import torch.nn as nn
import torch.nn.functional as F
class MinstConv(nn.Module):
    def __init__(self):
        super(MinstConv, self).__init__()
        #each model needs nn.module for quantization to work
        self.model = nn.ModuleDict(dict(
            conv1 = nn.Conv2d(1, 32, 3, 1),
            relu1 = nn.ReLU(),
            conv2 = nn.Conv2d(32, 64, 3, 1),
            relu2 = nn.ReLU(),
            maxpool2d = nn.MaxPool2d(kernel_size=2),
            dropout1 = nn.Dropout(0.25),
            flatten = nn.Flatten(),
            fc1 = nn.Linear(9216, 128),
            relu3 = nn.ReLU(),
            dropout2 = nn.Dropout(0.5),
            fc2 = nn.Linear(128, 10)
        ))


    def forward(self, x):
        output = x
        for layer_name, layer in self.model.items():
            output = layer(output)
        output = F.log_softmax(output, dim=1)
        return output
