# Copyright (c) Ye Liu. All rights reserved.

import os
from functools import wraps
from subprocess import getoutput

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nncore


def _init_dist_pytorch(backend, **kwargs):
    if torch.cuda.is_available():
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)


def _init_dist_slurm(backend, port=29500, **kwargs):
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = os.environ['SLURM_NTASKS']
    node_list = os.environ['SLURM_NODELIST']
    addr = getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_ADDR'] = addr
    os.environ['MASTER_PORT'] = str(port)
    os.environ['WORLD_SIZE'] = ntasks
    os.environ['RANK'] = str(proc_id)
    torch.cuda.set_device(proc_id % torch.cuda.device_count())
    dist.init_process_group(backend=backend, **kwargs)


def _get_default_device(group=None):
    backend = dist.get_backend(group)
    device = torch.device('cuda' if backend == 'nccl' else 'cpu')
    return device


def _serialize_to_tensor(data, device):
    buffer = nncore.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    data_tensor = torch.ByteTensor(storage).to(device)
    size_tensor = torch.LongTensor([data_tensor.numel()]).to(device)
    return data_tensor, size_tensor


def _pad_tensor(data_tensor, pad_size):
    data_size = data_tensor.numel()
    if data_size < pad_size:
        padding = data_tensor.new_empty(pad_size - data_size)
        data_tensor = torch.cat((data_tensor, padding))
    return data_tensor


def init_dist(launcher='pytorch', backend='gloo', **kwargs):
    """
    Initialize a distributed process group.

    Args:
        launcher (str, optional): Launcher for the process group. Currently
            supported launchers include ``pytorch`` and ``slurm``. Default:
            ``'pytorch'``.
        backend (:obj:`dist.Backend` or str, optional): The distribution
            backend to use. This field should be given as a :obj:`dist.Backend`
            object or a str (e.g. ``'gloo'``) which can also be accessed via
            :obj:`dist.Backend` attributes. Depending on build-time
            configurations, valid values include ``'gloo'`` and ``'nccl'``. If
            using multiple processes per machine with ``nccl`` backend, each
            process must have exclusive access to every GPU it uses, as
            sharing GPUs between processes can result in deadlocks. Default:
            ``'gloo'``.
    """
    assert backend in ('gloo', 'nccl')

    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    elif launcher == 'slurm':
        _init_dist_slurm(backend, **kwargs)
    else:
        raise TypeError("unsupported launcher: '{}'".format(launcher))


def is_distributed():
    """
    Check whether the current process is distributed.

    Returns:
        bool: Whether the current process is distributed.
    """
    return dist.is_available() and dist.is_initialized()


def get_rank(group=None):
    """
    Get the rank of the current process in a process group.

    Args:
        group (:obj:`dist.ProcessGroup` or None, optional): The process group
            to use. If not specified, the default process group will be used.
            Default: ``None``.

    Returns:
        int: The process rank.
    """
    if not is_distributed():
        return 0
    return dist.get_rank(group=group or dist.group.WORLD)


def get_world_size(group=None):
    """
    Get the world size of a process group.

    Args:
        group (:obj:`dist.ProcessGroup` or None, optional): The process group
            to use. If not specified, the default process group will be used.
            Default: ``None``.

    Returns:
        int: The world size.
    """
    if not is_distributed():
        return 1
    return dist.get_world_size(group=group or dist.group.WORLD)


def get_dist_info(group=None):
    """
    Get the rank of the current process and the world size of a process group.

    Args:
        group (:obj:`dist.ProcessGroup` or None, optional): The process group
            to use. If not specified, the default process group will be used.
            Default: ``None``.

    Returns:
        tuple[int]: The process rank and the world size.
    """
    if not is_distributed():
        return 0, 1
    group = group or dist.group.WORLD
    return get_rank(group=group), get_world_size(group=group)


def is_main_process():
    """
    Check whether the current process is the main process.

    Returns:
        bool: Whether the current process is the main process.
    """
    return get_rank() == 0


def synchronize(group=None):
    """
    Synchronize all processes in a process group.
    """
    if not dist.is_available() or not dist.is_initialized() or get_world_size(
            group=group) == 1:
        return
    dist.barrier(group=group)


def broadcast(data=None, src=0, group=None):
    """
    Perform :obj:`dist.broadcast` on arbitrary serializable data.

    Args:
        data (any, optional): Any serializable object.
        src (int, optional): The source rank. Default: ``0``.
        group (:obj:`dist.ProcessGroup` or None, optional): The process group
            to use. If not specified, the default process group will be used.
            Default: ``None``.

    Returns:
        any: The data broadcasted from the source rank.
    """
    rank, world_size = get_dist_info(group=group)
    if world_size == 1:
        return data

    device = _get_default_device(group=group)

    if rank == src:
        data_tensor, size_tensor = _serialize_to_tensor(data, device)
    else:
        size_tensor = torch.empty(1, dtype=torch.long, device=device)

    dist.broadcast(size_tensor, src=src, group=group)
    pad_size = size_tensor.item()

    if rank == src:
        data_tensor = _pad_tensor(data_tensor, pad_size)
    else:
        data_tensor = torch.empty(pad_size, dtype=torch.uint8, device=device)

    dist.broadcast(data_tensor, src=src, group=group)
    buffer = data_tensor.cpu().numpy().tobytes()[:pad_size]
    broadcasted = nncore.loads(buffer)

    return broadcasted


def all_gather(data, group=None):
    """
    Perform :obj:`dist.all_gather` on arbitrary serializable data.

    Args:
        data (any): Any serializable object.
        group (:obj:`dist.ProcessGroup` or None, optional): The process group
            to use. If not specified, the default process group will be used.
            Default: ``None``.

    Returns:
        list: The list of data gathered from each rank.
    """
    world_size = get_world_size(group=group)
    if world_size == 1:
        return [data]

    device = _get_default_device(group=group)
    data_tensor, size_tensor = _serialize_to_tensor(data, device)
    size_list = [torch.empty_like(size_tensor) for _ in range(world_size)]
    dist.all_gather(size_list, size_tensor, group=group)

    pad_size = max(size_tensor.item() for size_tensor in size_list)
    data_tensor = _pad_tensor(data_tensor, pad_size)

    tensor_list = [data_tensor.new_empty(pad_size) for _ in range(world_size)]
    dist.all_gather(tensor_list, data_tensor, group=group)

    gathered = []
    for data_tensor, size_tensor in zip(tensor_list, size_list):
        buffer = data_tensor.cpu().numpy().tobytes()[:size_tensor.item()]
        gathered.append(nncore.loads(buffer))

    return gathered


def gather(data, dst=0, group=None):
    """
    Perform :obj:`dist.gather` on arbitrary serializable data.

    Args:
        data (any): Any serializable object.
        dst (int, optional): The destination rank. Default: ``0``.
        group (:obj:`dist.ProcessGroup` or None, optional): The process group
            to use. If not specified, the default process group will be used.
            Default: ``None``.

    Returns:
        list or ``None``: On ``dst``, it should be a list of data gathered \
            from each rank. Otherwise, ``None``.
    """
    rank, world_size = get_dist_info(group=group)
    if world_size == 1:
        return [data]

    device = _get_default_device(group=group)
    data_tensor, size_tensor = _serialize_to_tensor(data, device)
    size_list = [torch.empty_like(size_tensor) for _ in range(world_size)]
    dist.all_gather(size_list, size_tensor, group=group)

    pad_size = max(size_tensor.item() for size_tensor in size_list)
    data_tensor = _pad_tensor(data_tensor, pad_size)

    if rank == dst:
        tensor_list = [
            data_tensor.new_empty(pad_size) for _ in range(world_size)
        ]
        dist.gather(data_tensor, gather_list=tensor_list, dst=dst, group=group)

        gathered = []
        for data_tensor, size_tensor in zip(tensor_list, size_list):
            buffer = data_tensor.cpu().numpy().tobytes()[:size_tensor.item()]
            gathered.append(nncore.loads(buffer))
    else:
        dist.gather(data_tensor, dst=dst, group=group)
        gathered = None

    return gathered


def master_only(func):
    """
    A decorator that makes a function can only be executed in the main process.
    """

    @wraps(func)
    def _wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return _wrapper
