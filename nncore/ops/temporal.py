# Copyright (c) Ye Liu. All rights reserved.

import torch


def temporal_area(windows):
    """
    Compute the areas of temporal windows.

    Args:
        windows (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed. They
            are expected to be in ``(start, end)`` format.

    Returns:
        :obj:`nn.Tensor[N]`: The computed areas.
    """
    return windows[:, 1] - windows[:, 0]


def temporal_intersection(windows1, windows2, aligned=False):
    """
    Compute the intersections among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the intersections
            among aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed \
            intersection values.
    """
    if aligned:
        s = torch.max(windows1[:, 0], windows2[:, 0])
        e = torch.min(windows1[:, 1], windows2[:, 1])
    else:
        s = torch.max(windows1[:, None, 0], windows2[:, 0])
        e = torch.min(windows1[:, None, 1], windows2[:, 1])

    inter = (e - s).clamp(0)
    return inter


def temporal_iou(windows1, windows2, aligned=False):
    """
    Compute the intersection-over-unions (IoUs) among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the IoU among
            aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed pairwise \
            IoU values.
    """
    area1 = temporal_area(windows1)
    area2 = temporal_area(windows2)

    inter = temporal_intersection(windows1, windows2, aligned=aligned)

    if aligned:
        iou = inter / (area1 + area2 - inter)
    else:
        iou = inter / (area1[:, None] + area2 - inter)

    return iou


def temporal_iof(windows1, windows2, aligned=False):
    """
    Compute the intersection-over-foregrounds (IoFs) among temporal windows.

    Args:
        windows1 (:obj:`nn.Tensor[N, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        windows2 (:obj:`nn.Tensor[M, 2]`): Temporal windows to be computed.
            They are expected to be in ``(start, end)`` format.
        aligned (bool, optional): Whether to only compute the IoF among
            aligned temporal windows. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N]` | :obj:`nn.Tensor[N, M]`: The computed pairwise \
            IoF values.
    """
    area_forground = temporal_area(windows1)

    inter = temporal_intersection(windows1, windows2, aligned=aligned)

    if aligned:
        iof = inter / area_forground
    else:
        iof = inter / area_forground[:, None]

    return iof
