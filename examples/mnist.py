# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from nncore.engine import Engine


class LeNet(nn.Module):

    def __init__(self):
        super(LeNet, self).__init__()

        # yapf:disable
        self.convs = nn.Sequential(
            nn.Conv2d(1, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2),
            nn.Conv2d(16, 120, 5))
        self.fcs = nn.Sequential(
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, 10))
        # yapf:enable

        self.loss = nn.CrossEntropyLoss()

    def forward(self, data, **kwargs):
        x, y = data[0], data[1]

        x = self.convs(x)
        x = x.squeeze()
        x = self.fcs(x)

        pred = torch.argmax(x, dim=1)
        acc = torch.eq(pred, y).sum().float() / x.size(0)
        loss = self.loss(x, y)

        return dict(_num_samples=x.size(0), acc=acc, loss=loss)


def main():
    # Prepare datasets and the model
    transform = Compose([ToTensor(), Resize(32), Normalize(0.5, 0.5)])

    train = MNIST('data', train=True, transform=transform, download=True)
    train_loader = DataLoader(train, batch_size=16, shuffle=True)

    val = MNIST('data', train=False, transform=transform, download=True)
    val_loader = DataLoader(val, batch_size=64, shuffle=False)

    data_loaders = dict(train=train_loader, val=val_loader)
    model = LeNet()

    # Initialize and launch engine
    engine = Engine(model, data_loaders)
    engine.launch()


if __name__ == '__main__':
    main()
