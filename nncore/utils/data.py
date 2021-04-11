# Copyright (c) Ye Liu. All rights reserved.

from collections.abc import Sequence
from itertools import chain

import numpy as np


def swap_element(matrix, i, j, dim=0):
    """
    Swap two elements of an array or a tensor.

    Args:
        matrix (:obj:`np.ndarray` or :obj:`torch.Tensor`): The array or tensor
            to be swapped.
        i (int or tuple): Index of the first element.
        j (int or tuple): Index of the second element.
        dim (int, optional): The dimension to swap. Default: ``0``.

    Returns:
        :obj:`np.ndarray` or :obj:`torch.Tensor`: The swapped array or tensor.
    """
    inds = [slice(0, matrix.shape[d]) for d in range(dim)]

    i_inds = inds + [i]
    j_inds = inds + [j]

    meth = 'copy' if isinstance(matrix, np.ndarray) else 'clone'
    m_i = getattr(matrix[i_inds], meth)()
    m_j = getattr(matrix[j_inds], meth)()

    matrix[i_inds] = m_j
    matrix[j_inds] = m_i

    return matrix


def is_seq_of(seq, expected_type, seq_type=Sequence):
    """
    Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Default:
            ``Sequence``.

    Returns:
        bool: Whether the sequence is valid.
    """
    if not isinstance(seq, seq_type):
        return False

    for item in seq:
        if not isinstance(item, expected_type):
            return False

    return True


def is_list_of(seq, expected_type):
    """
    Check whether it is a list of some type.

    A partial method of :obj:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=list)


def is_tuple_of(seq, expected_type):
    """
    Check whether it is a tuple of some type.

    A partial method of :obj:`is_seq_of`.
    """
    return is_seq_of(seq, expected_type, seq_type=tuple)


def slice_list(in_list, length):
    """
    Slice a list into several sub lists by length.

    Args:
        in_list (list): The list to be sliced.
        length (list[int] or int): The expected length or list of lengths of
            output lists.

    Returns:
        list[list]: The sliced lists.
    """
    if isinstance(length, int):
        assert len(in_list) % length == 0
        length = [length] * int(len(in_list) / length)
    elif not isinstance(length, list):
        raise TypeError("'length' must be an integer or a list of integers")
    elif sum(length) != len(in_list):
        raise ValueError('sum of the length and the list length are mismatch')

    out_list, idx = [], 0
    for i in range(len(length)):
        out_list.append(in_list[idx:idx + length[i]])
        idx += length[i]

    return out_list


def concat_list(in_list):
    """
    Concatenate a list of lists into a single list.

    Args:
        in_list (list): The list of lists to be merged.

    Returns:
        list: The concatenated flat list.
    """
    return list(chain(*in_list))


def to_dict_of_list(in_list):
    """
    Convert a list of dicts to a dict of lists.

    Args:
        in_list (list): The list of dicts to be converted.

    Returns:
        dict: The converted dict of lists.
    """
    for i in range(len(in_list) - 1):
        if in_list[i].keys() != in_list[i + 1].keys():
            raise ValueError('dict keys are not consistent')

    out_dict = dict()
    for key in in_list[0]:
        out_dict[key] = [item[key] for item in in_list]

    return out_dict


def to_list_of_dict(in_dict):
    """
    Convert a dict of lists to a list of dicts.

    Args:
        in_dict (dict): the dict of lists to be converted.

    Returns:
        list: The converted list of dicts.
    """
    values = in_dict.values()
    for i in range(len(in_dict) - 1):
        if len(values[i]) != len(values[i + 1]):
            raise ValueError('lengths of lists are not consistent')

    out_list = []
    for i in range(len(in_dict)):
        out_list.append({k: v[i] for k, v in in_dict.items()})

    return out_list
