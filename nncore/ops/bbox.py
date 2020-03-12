# Copyright (c) Ye Liu. All rights reserved.

import torch


def bbox_area(bboxes):
    """
    Compute the area of a set of bounding boxes, which are specified by its
    (x1, y1, x2, y2) coordinates.

    Args:
        bboxes (Tensor[N, 4]): bboxes for which the area will be computed. They
            are expected to be in (x1, y1, x2, y2) format.

    Returns:
        areas (Tensor[N]): area for each box
    """
    return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])


def bbox_intersection(bboxes1, bboxes2, aligned=False):
    """
    Compute intersection of bboxes.

    Args:
        bboxes1 (Tensor[N, 4]): bboxes to be computed. Expected to be in
            (x1, y1, x2, y2) format
        bboxes2 (Tensor[M, 4]): bboxes to be computed. Expected to be in
            (x1, y1, x2, y2) format
        aligned (bool, optional): whether to only compute the intersection of
            the aligned bboxes

    Returns:
        inter (Tensor[N, M] or Tensor[N, 1]): the tensor containing the
            intersection values
    """
    if aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, 0] * wh[:, 1]
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])

        wh = (rb - lt).clamp(min=0)
        inter = wh[:, :, 0] * wh[:, :, 1]

    return inter


def bbox_iou(bboxes1, bboxes2, aligned=False):
    """
    Compute intersection-over-union (Jaccard index) of bboxes.

    Args:
        bboxes1 (Tensor[N, 4]): bboxes to be computed. Expected to be in
            (x1, y1, x2, y2) format
        bboxes2 (Tensor[M, 4]): bboxes to be computed. Expected to be in
            (x1, y1, x2, y2) format
        aligned (bool, optional): whether to only compute the IoU of the
            aligned bboxes

    Returns:
        iou (Tensor[N, M] or Tensor[N, 1]): the tensor containing the pairwise
            IoU values
    """
    area1 = bbox_area(bboxes1)
    area2 = bbox_area(bboxes2)

    inter = bbox_intersection(bboxes1, bboxes2, aligned=aligned)
    iou = inter / (area1[:, None] + area2 - inter)

    return iou


def bbox_iof(bboxes1, bboxes2, aligned=False):
    """
    Compute intersection-over-foreground (Jaccard index) of bboxes.

    Args:
        bboxes1 (Tensor[N, 4]): bboxes to be computed. Expected to be in
            (x1, y1, x2, y2) format.
        bboxes2 (Tensor[M, 4]): bboxes to be computed. Expected to be in
            (x1, y1, x2, y2) format.
        aligned (bool, optional): whether to only compute the IoF of the
            aligned bboxes

    Returns:
        iof (Tensor[N, M] or Tensor[N, 1]): the tensor containing the pairwise
            IoF values
    """
    area_forground = bbox_area(bboxes1)

    inter = bbox_intersection(bboxes1, bboxes2, aligned=aligned)
    iof = inter / area_forground[:, None]

    return iof


def remove_small_bboxes(bboxes, min_size):
    """
    Remove bboxes which contains at least one side smaller than min_size.

    Args:
        bboxes (Tensor[N, 4]): bboxes in (x1, y1, x2, y2) format
        min_size (float): the minimum size

    Returns:
        keep (Tensor[K]): indices of the bboxes that have both sides larger
            than min_size
    """
    ws, hs = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]

    keep = (ws >= min_size) & (hs >= min_size)
    keep = keep.nonzero().squeeze(1)

    return keep
