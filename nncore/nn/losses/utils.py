# Copyright (c) Ye Liu. Licensed under the MIT License.

from functools import wraps


def weighted_loss(func):
    """
    A syntactic sugar for loss functions with dynamic weights and average
    factors. This method is expected to be used as a decorator.
    """

    @wraps(func)
    def _wrapper(*args,
                 weight=None,
                 reduction='mean',
                 avg_factor=None,
                 **kwargs):
        assert reduction in ('mean', 'sum', 'none')
        loss = func(*args, **kwargs)

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
