# Copyright (c) Ye Liu. All rights reserved.

from torch.nn.parallel import DataParallel, DistributedDataParallel

from .utils import scatter_kwargs


class NNDataParallel(DataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)


class NNDistributedDataParallel(DistributedDataParallel):

    def scatter(self, inputs, kwargs, device_ids):
        return scatter_kwargs(inputs, kwargs, device_ids, dim=self.dim)
