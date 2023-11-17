import os
import torch
import torch.multiprocessing as mp
from torch import distributed as dist

def init_dist(launcher, backend='nccl', **kwargs):
    """
    Initialize the distributed environment.

    Parameters:
    - launcher (str): The type of launcher to use for initializing the distributed environment.
    - backend (str, optional): The backend to use for distributed computing. Defaults to 'nccl'.
    - **kwargs: Additional keyword arguments.

    Raises:
    - ValueError: If an invalid launcher type is provided.
    """
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')
    if launcher == 'pytorch':
        _init_dist_pytorch(backend, **kwargs)
    else:
        raise ValueError(f'Invalid launcher type: {launcher}')


def _init_dist_pytorch(backend, **kwargs):
    """
    Initialize the PyTorch distributed process group.

    This is an internal function called by init_dist when the launcher is set to 'pytorch'.

    Parameters:
    - backend (str): The backend to use for distributed computing.
    - **kwargs: Additional keyword arguments for process group initialization.
    """
    # TODO: use local_rank instead of rank % num_gpus
    rank = int(os.environ['RANK'])
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(rank % num_gpus)
    dist.init_process_group(backend=backend, **kwargs)
    print(f'init distributed in rank {torch.distributed.get_rank()}')


def get_dist_info():
    """
    Get the rank and world size in the distributed environment.

    Returns:
    - rank (int): The rank of the current process.
    - world_size (int): The total number of processes in the distributed environment.
    """
    if dist.is_available():
        initialized = dist.is_initialized()
    else:
        initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def reduce_tensor(data, world_size):
    """
    Reduce a tensor across all processes to calculate the mean.

    Parameters:
    - data (torch.Tensor or dict or list): The data to be reduced. Can be a tensor, a dictionary of tensors, or a list of tensors.
    - world_size (int): The total number of processes in the distributed environment.

    Returns:
    - Reduced tensor, dictionary, or list, averaged across all processes.
    """
    def _reduce_tensor(tensor):
        if isinstance(tensor, dict):
            return {k: _reduce_tensor(v) for k, v in tensor.items()}
        elif isinstance(tensor, list):
            return [_reduce_tensor(v) for v in tensor]
        elif isinstance(tensor, torch.Tensor):
            rt = tensor.clone()
            torch.distributed.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
            rt /= world_size
            return rt

    return _reduce_tensor(data)

def gather_tensor(tensor, args):
    """
    Gather tensors from all processes and concatenate them.

    Parameters:
    - tensor (torch.Tensor): The tensor to gather.
    - args: Arguments containing 'world_size', the total number of processes in the distributed environment.

    Returns:
    - A concatenated tensor containing data from all processes.
    """
    output_tensors = [tensor.clone() for _ in range(args.world_size)]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    return concat
