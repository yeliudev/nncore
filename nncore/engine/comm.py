# Copyright (c) Ye Liu. All rights reserved.

import os
from functools import wraps

import pynvml
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import nncore


def init_dist(backend, **kwargs):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    if torch.cuda.is_available():
        rank = int(os.environ['RANK'])
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(rank % num_gpus)

    dist.init_process_group(backend=backend, **kwargs)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank(group=dist.group.WORLD):
    return dist.get_rank(group=group) if is_distributed() else 0


def get_world_size(group=dist.group.WORLD):
    return dist.get_world_size(group=group) if is_distributed() else 1


def get_dist_info(group=dist.group.WORLD):
    return (get_rank(group=group),
            get_world_size(group=group)) if is_distributed() else (0, 1)


def is_main_process():
    return get_rank() == 0


def synchronize(group=dist.group.WORLD):
    """
    Synchronize among all processes in a process group.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if get_world_size(group=group) == 1:
        return
    dist.barrier(group=group)


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ['nccl', 'gloo']
    device = torch.device('cuda' if backend == 'nccl' else 'cpu')

    if backend == 'nccl':
        world_size = get_world_size()
        total_size = len(bytearray(nncore.dumps(data))) * world_size

        pynvml.nvmlInit()
        for i in range(world_size):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            if meminfo.free < total_size:
                group = dist.new_group(backend='gloo')
                device = 'cpu'
                break

    buffer = nncore.dumps(data)
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)

    return tensor, group


def _pad_tensors(tensor, group):
    world_size = get_world_size(group=group)
    local_size = torch.LongTensor([tensor.numel()], device=tensor.device)
    size_list = [local_size.clone() for _ in range(world_size)]
    dist.all_gather(size_list, local_size, group=group)

    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    if local_size != max_size:
        padding = torch.ByteTensor([max_size - local_size],
                                   device=tensor.device)
        tensor = torch.cat((tensor, padding))

    return size_list, tensor


def all_gather(data, group=dist.group.WORLD):
    """
    Run all_gather on arbitrary serializable data.

    Args:
        data (any): any serializable object
        group (ProcessGroup, optional): a torch process group

    Returns:
        gathered (list[data]): a list of data gathered from each rank
    """
    if get_world_size(group=group) == 1:
        return [data]

    tensor, group = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_tensors(tensor, group)
    max_size = max(size_list)

    tensor_list = [tensor.new_empty([max_size]) for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    gathered = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        gathered.append(nncore.loads(buffer))

    return gathered


def gather(data, dst=0, group=dist.group.WORLD):
    """
    Run gather on arbitrary serializable data.

    Args:
        data (any): any serializable object
        dst (int): destination rank
        group (ProcessGroup, optional): a torch process group

    Returns:
        gathered (list[data]) or None: on dst, a list of data gathered from
            each rank. Otherwise, None.
    """
    rank, world_size = get_dist_info(group=group)
    if get_world_size() == 1:
        return [data]

    tensor, group = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_tensors(tensor, group)

    if rank == dst:
        max_size = max(size_list)
        tensor_list = [tensor.new_empty([max_size]) for _ in size_list]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        gathered = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            gathered.append(nncore.loads(buffer))
    else:
        dist.gather(tensor, dst=dst, group=group)
        gathered = None

    return gathered


def master_only(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        if is_main_process():
            return func(*args, **kwargs)

    return wrapper
