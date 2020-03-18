# Copyright (c) Ye Liu. All rights reserved.

import torch
import torchvision
import torch.nn as nn

import nncore
from nncore.engine import Engine


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        # yapf:disable
        self.convs = nn.Sequential(
            nn.Conv2d(1, 6, 5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2))
        self.fcs = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(inplace=True),
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
            nn.Linear(84, 10))
        # yapf:enable

        self.criterion = nn.CrossEntropyLoss()

    def forward(self, data, **kwargs):
        x, labels = data
        x = self.convs(x)
        x = x.view(x.size()[0], -1)
        x = self.fcs(x)
        loss = self.criterion(x, labels)
        return dict(loss=loss)


def main():
    cfg = nncore.Config.from_file('examples/config.py')

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.5], [0.5])
    ])

    trainset = torchvision.datasets.MNIST(
        'data', train=True, transform=transform, download=True)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=16, shuffle=True, num_workers=2)

    testset = torchvision.datasets.MNIST(
        'data', train=False, transform=transform, download=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=16, shuffle=False, num_workers=2)

    data_loaders = dict(train=trainloader, val=testloader)
    model = LeNet()

    engine = Engine(model, data_loaders, **cfg)
    engine.launch()


if __name__ == '__main__':
    main()
