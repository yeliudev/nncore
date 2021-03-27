# Copyright (c) Ye Liu. All rights reserved.

import torch


def bbox_area(bboxes):
    """
    Compute the areas of bounding boxes.

    Args:
        bboxes (:obj:`nn.Tensor[N, 4]`): Bounding boxes to be computed. They
            are expected to be in ``(x1, y1, x2, y2)`` format.

    Returns:
        :obj:`nn.Tensor[N]`: The computed areas.
    """
    return (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])


def bbox_intersection(bboxes1, bboxes2, aligned=False):
    """
    Compute the intersections among bounding boxes.

    Args:
        bboxes1 (:obj:`nn.Tensor[N, 4]`): Bounding boxes to be computed. They
            are expected to be in ``(x1, y1, x2, y2)`` format.
        bboxes2 (:obj:`nn.Tensor[M, 4]`): Bounding boxes to be computed. They
            are expected to be in ``(x1, y1, x2, y2)`` format.
        aligned (bool, optional): Whether to only compute the intersections
            among aligned bounding boxes. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N, M]`: The computed intersection values.
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
    Compute the intersection-over-unions (IoUs) among bounding boxes.

    Args:
        bboxes1 (:obj:`nn.Tensor[N, 4]`): Bounding boxes to be computed. They
            are expected to be in ``(x1, y1, x2, y2)`` format.
        bboxes2 (:obj:`nn.Tensor[M, 4]`): Bounding boxes to be computed. They
            are expected to be in ``(x1, y1, x2, y2)`` format.
        aligned (bool, optional): Whether to only compute the IoU among
            aligned bounding boxes. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N, M]`: The computed pairwise IoU values.
    """
    area1 = bbox_area(bboxes1)
    area2 = bbox_area(bboxes2)

    inter = bbox_intersection(bboxes1, bboxes2, aligned=aligned)
    iou = inter / (area1[:, None] + area2 - inter)

    return iou


def bbox_iof(bboxes1, bboxes2, aligned=False):
    """
    Compute the intersection-over-foregrounds (IoFs) among bounding boxes.

    Args:
        bboxes1 (:obj:`nn.Tensor[N, 4]`): Bounding boxes to be computed. They
            are expected to be in ``(x1, y1, x2, y2)`` format.
        bboxes2 (:obj:`nn.Tensor[M, 4]`): Bounding boxes to be computed. They
            are expected to be in ``(x1, y1, x2, y2)`` format.
        aligned (bool, optional): Whether to only compute the IoF among
            aligned bounding boxes. Default: ``False``.

    Returns:
        :obj:`nn.Tensor[N, M]`: The computed pairwise IoF values.
    """
    area_forground = bbox_area(bboxes1)

    inter = bbox_intersection(bboxes1, bboxes2, aligned=aligned)
    iof = inter / area_forground[:, None]

    return iof


def remove_small_bboxes(bboxes, min_size):
    """
    Remove bounding boxes which contains at least one side smaller than
    the minimum size.

    Args:
        bboxes (:obj:`nn.Tensor[N, 4]`): Bounding boxes to be computed. They
            are expected to be in ``(x1, y1, x2, y2)`` format.
        min_size (float): The minimum size of bounding boxes.

    Returns:
        :obj:`nn.Tensor[K]`: Indices of the bounding boxes that have both \
            sides larger than ``min_size``.
    """
    ws, hs = bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1]

    keep = (ws >= min_size) & (hs >= min_size)
    keep = keep.nonzero(as_tuple=False).squeeze(1)

    return keep
