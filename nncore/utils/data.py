# Copyright (c) Ye Liu. All rights reserved.

import numpy as np


def swap_element(matrix, i, j, dim=0):
    """
    Swap two elements of an array or a tensor.

    Args:
        matrix (:obj:`np.ndarray` | :obj:`torch.Tensor`): The array or tensor
            to be swapped.
        i (int | tuple): Index of the first element.
        j (int | tuple): Index of the second element.
        dim (int, optional): The dimension to swap. Default: ``0``.

    Returns:
        :obj:`np.ndarray` | :obj:`torch.Tensor`: The swapped array or tensor.
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


def is_seq_of(seq, item_type, seq_type=(list, tuple)):
    """
    Check whether it is a sequence of some type.

    Args:
        seq (Sequence): The sequence to be checked.
        item_type (tuple[type] | type): Expected item type.
        seq_type (tuple[type] | type, optional): Expected sequence type.
            Default: ``(list, tuple)``.

    Returns:
        bool: Whether the sequence is valid.
    """
    if not isinstance(seq, seq_type):
        return False

    for item in seq:
        if not isinstance(item, item_type):
            return False

    return True


def is_list_of(seq, item_type):
    """
    Check whether it is a list of some type.

    A partial method of :obj:`is_seq_of`.
    """
    return is_seq_of(seq, item_type, seq_type=list)


def is_tuple_of(seq, item_type):
    """
    Check whether it is a tuple of some type.

    A partial method of :obj:`is_seq_of`.
    """
    return is_seq_of(seq, item_type, seq_type=tuple)


def slice(seq, length, type='list'):
    """
    Slice a sequence into several sub sequences by length.

    Args:
        seq (list | tuple): The sequence to be sliced.
        length (list[int] | int): The expected length or list of lengths.
        type (str, optional): The type of returned object. Expected values
            include ``'list'`` and ``'tuple'``. Default: ``'list'``.

    Returns:
        list[list]: The sliced sequences.
    """
    assert type in ('list', 'tuple')

    if isinstance(length, int):
        assert len(seq) % length == 0
        length = [length] * int(len(seq) / length)
    elif not isinstance(length, list):
        raise TypeError("'length' must be an integer or a list of integers")
    elif sum(length) != len(seq):
        raise ValueError('the total length do not match the sequence length')

    out, idx = [], 0
    for i in range(len(length)):
        out.append(seq[idx:idx + length[i]])
        idx += length[i]

    if type == 'tuple':
        out = tuple(out)

    return out


def concat(seq):
    """
    Concatenate a sequence of sequences.

    Args:
        seq (list | tuple): The sequence to be concatenated.

    Returns:
        list | tuple: The concatenated sequence.
    """
    seq_type = type(seq)

    out = []
    for item in seq:
        out += item

    return seq_type(out)


def flatten(seq):
    """
    Flatten a sequence of sequences and items.

    Args:
        seq (list | tuple): The sequence to be flattened.

    Returns:
        list | tuple: The flattened sequence.
    """
    seq_type = type(seq)

    out = []
    for item in seq:
        if isinstance(item, (list, tuple)):
            out += flatten(item)
        else:
            out.append(item)

    return seq_type(out)


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
