# Copyright (c) Ye Liu. All rights reserved.

import os.path as osp
from shutil import rmtree
from tempfile import mkdtemp

import pynvml
import torch
import torch.distributed as dist

import nncore


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def get_rank():
    return dist.get_rank() if is_distributed() else 0


def get_world_size():
    return dist.get_world_size() if is_distributed() else 1


def get_dist_info():
    return (get_rank(), get_world_size()) if is_distributed() else (0, 1)


def is_main_process():
    return get_rank() == 0


def synchronize():
    """
    Synchronizes among all processes when using distributed training.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    if get_world_size() == 1:
        return
    dist.barrier()


def all_gather(data):
    """
    Run all_gather on arbitrary serializable data.

    Args:
        data (any): any serializable object

    Returns:
        collected (list[data]): a list of data gathered from each rank
    """
    world_size = get_world_size()
    total_size = len(bytearray(nncore.dumps(data))) * world_size

    pynvml.nvmlInit()
    matched = False
    for i in range(world_size):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        if meminfo.free < total_size:
            matched = True
            break

    if matched:
        return _all_gather_cpu(data)
    else:
        return _all_gather_gpu(data)


def _all_gather_gpu(data):
    world_size = get_world_size()

    part_tensor = torch.cuda.ByteTensor(bytearray(nncore.dumps(data)))
    size_tensor = torch.cuda.LongTensor([part_tensor.shape[0]])
    size_list = [size_tensor.clone() for _ in range(world_size)]
    dist.all_gather(size_list, size_tensor)

    max_size = torch.LongTensor(size_list).max()
    part_send = torch.zeros(max_size, dtype=torch.uint8, device='cuda')
    part_send[:size_tensor[0]] = part_tensor
    recv_list = [part_tensor.new_zeros(max_size) for _ in range(world_size)]
    dist.all_gather(recv_list, part_send)

    if is_main_process():
        collected = []
        for recv, size in zip(recv_list, size_list):
            collected.append(
                nncore.loads(recv[:size.item()].cpu().numpy().tobytes()))
    else:
        collected = None

    return collected


def _all_gather_cpu(data):
    rank, world_size = get_dist_info()
    dir_tensor = torch.full((512, ), 32, dtype=torch.uint8, device='cuda')

    if is_main_process():
        tmpdir = torch.cuda.ByteTensor(bytearray(mkdtemp().encode()))
        dir_tensor[:len(tmpdir)] = tmpdir

    dist.broadcast(dir_tensor, 0)
    tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    nncore.dump(data, osp.join(tmpdir, 'part_{}.pkl'.format(rank)))
    synchronize()

    if is_main_process():
        collected = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, 'part_{}.pkl'.format(i))
            collected.append(nncore.load(part_file))
        rmtree(tmpdir)
    else:
        collected = None

    return collected
