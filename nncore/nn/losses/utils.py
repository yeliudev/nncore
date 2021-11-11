# Copyright (c) Ye Liu. All rights reserved.

from functools import wraps


def weighted_loss(func):
    """
    A syntactic sugar for loss functions with dynamic weights and average
    factors. This method is expected to be used as a decorator.
    """

    @wraps(func)
    def _wrapper(pred,
                 target,
                 weight=None,
                 reduction='mean',
                 avg_factor=None,
                 **kwargs):
        assert reduction in ('mean', 'sum', 'none')
        loss = func(pred, target, **kwargs)

        if weight is not None:
            loss = loss * weight

        if reduction == 'mean':
            if avg_factor is None:
                loss = loss.mean()
            else:
                loss = loss.sum() / avg_factor
        elif reduction == 'sum':
            loss = loss.sum()

        return loss

    return _wrapper
