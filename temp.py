from torch import nn

test = nn.ModuleDict(dict(
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

print(test["conv2"])
test.add_module("conv2", nn.Conv2d(34, 70, 8, 3))

print(test["conv2"])