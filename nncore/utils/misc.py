# Copyright (c) Ye Liu. All rights reserved.

from collections.abc import Iterable, Sequence
from itertools import chain

import numpy as np


def swap_element(matrix, i, j):
    """
    Swap two elements of an array or a tensor.

    Args:
        matrix (:obj:`np.ndarray` or :obj:`torch.Tensor`): The array or tensor
            to be swapped.
        i (int): Index of the first element.
        j (int): Index of the second element.

    Returns:
        :obj:`np.ndarray` or :obj:`torch.Tensor`: The swapped array or tensor.
    """
    if isinstance(matrix, np.ndarray):
        tmp = matrix[i].copy()
    else:
        tmp = matrix[i].clone()

    matrix[i] = matrix[j]
    matrix[j] = tmp

    return matrix


def iter_cast(inputs, dst_type, return_type=None):
    """
    Cast elements of an iterable object into some type.

    Args:
        inputs (Iterable): The input iterable object.
        dst_type (type): Destination type.
        return_type (type, optional): The type of returned object. If
            specified, the output object will be converted to this type,
            otherwise an iterator. Default: ``None``.

    Returns:
        iterator or specified type: The converted object.
    """
    if not isinstance(inputs, Iterable):
        raise TypeError('inputs must be an iterable object')
    if not isinstance(dst_type, type):
        raise TypeError("'dst_type' must be a valid type")

    out_iter = map(dst_type, inputs)

    if return_type is not None:
        out_iter = return_type(out_iter)

    return out_iter


def list_cast(inputs, dst_type):
    """
    Cast elements of an iterable object into a list of some type.

    A partial method of :obj:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=list)


def tuple_cast(inputs, dst_type):
    """
    Cast elements of an iterable object into a tuple of some type.

    A partial method of :obj:`iter_cast`.
    """
    return iter_cast(inputs, dst_type, return_type=tuple)


def is_seq_of(seq, expected_type, seq_type=None):
    """
    Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        expected_type (type): Expected type of sequence items.
        seq_type (type, optional): Expected sequence type. Default: ``None``.

    Returns:
        bool: Whether the sequence is valid.
    """
    if seq_type is None:
        exp_seq_type = Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
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


def slice_list(in_list, lens):
    """
    Slice a list into several sub lists by a list of given length.

    Args:
        in_list (list): The list to be sliced.
        lens (list or int): The expected length of each out list.

    Returns:
        list: The sliced lists.
    """
    if isinstance(lens, int):
        assert len(in_list) % lens == 0
        lens = [lens] * int(len(in_list) / lens)
    if not isinstance(lens, list):
        raise TypeError("'indices' must be an integer or a list of integers")
    elif sum(lens) != len(in_list):
        raise ValueError(
            'sum of lens and list length does not match: {} != {}'.format(
                sum(lens), len(in_list)))
    out_list = []
    idx = 0
    for i in range(len(lens)):
        out_list.append(in_list[idx:idx + lens[i]])
        idx += lens[i]
    return out_list


def concat_list(in_list):
    """
    Concatenate a list of list into a single list.

    Args:
        in_list (list): The list of list to be merged.

    Returns:
        list: The concatenated flat list.
    """
    return list(chain(*in_list))


def to_dict_of_list(in_list):
    """
    Convert a list of dict to a dict of list.

    Args:
        in_list (list): The list of dict to be converted.

    Returns:
        dict: The converted dict of list.
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
    Convert a dict of list to a list of dict.

    Args:
        in_dict (dict): the dict of list to be converted.

    Returns:
        list: The converted list of dict.
    """
    values = in_dict.values()
    for i in range(len(in_dict) - 1):
        if len(values[i]) != len(values[i + 1]):
            raise ValueError('lengths of lists are not consistent')

    out_list = []
    for i in range(len(in_dict)):
        out_list.append({k: v[i] for k, v in in_dict.items()})

    return out_list
