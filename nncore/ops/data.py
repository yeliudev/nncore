# Copyright (c) Ye Liu. All rights reserved.

import torch
import torch.nn.functional as F


def cosine_similarity(x, y):
    """
    Compute the cosine similarities among two batches of vectors.

    Args:
        x (:obj:`torch.Tensor[N, F]`): The first batch of vectors.
        y (:obj:`torch.Tensor[M, F]`): The second batch of vectors.

    Returns:
        :obj:`torch.Tensor[N, M]`: The computed pairwise cosine similarities.
    """
    x = F.normalize(x)
    y = F.normalize(y)
    return torch.mm(x, y.t())
