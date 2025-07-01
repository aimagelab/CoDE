# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
"""
Misc functions, including distributed helpers.

Mostly copy-paste from torchvision references.
"""
import os
import platform
import random
import socket
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import distributed as dist
import cv2  # must come after the import of torch
import datetime
# --------------------------
DEBUG = os.getenv("DEBUG", "False").lower() in ("true", "1", "t")
DEBUG_BATCHES = 200


# --------------------------

def _find_free_port() -> str:
    # Copied from https://github.com/facebookresearch/detectron2/blob/main/detectron2/engine/launch.py # noqa: E501
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(('', 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _is_free_port(port: int) -> bool:
    ips = socket.gethostbyname_ex(socket.gethostname())[-1]
    ips.append('localhost')
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return all(s.connect_ex((ip, port)) != 0 for ip in ips)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__

    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop("force", False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def save_on_master(*args, **kwargs):
    if is_main_process():
        torch.save(*args, **kwargs)


def set_random_seed(seed, deterministic=False, benchmark=False):
    """Set random seed.
    Args:
        seed (int): Seed to be used.
        deterministic (bool): Whether to set the deterministic option for
            CUDNN backend, i.e., set `torch.backends.cudnn.deterministic`
            to True and `torch.backends.cudnn.benchmark` to False.
            Default: False.
    """
    # fix the seed for reproducibility
    seed = seed + get_rank()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = benchmark  # Dai dai dai!
    if deterministic:
        torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False


def init_distributed_mode_pytorch(backend="nccl", **kwargs):
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method("spawn")

    print("Using DDP ENV mode")
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    gpu = int(os.environ["LOCAL_RANK"])

    dist_url = "env://"

    torch.cuda.set_device(gpu)

    print(
        "|| dist_backend={} | world_size={} | global_rank={} | local_rank={} | dist_url={}".format(
            backend, world_size, rank, gpu, dist_url
        ),
        flush=True,
    )
    if "OMP_NUM_THREADS" in os.environ:
        print(f"OMP_NUM_THREADS={os.environ['OMP_NUM_THREADS']}")

    torch.distributed.init_process_group(
        backend=backend,
        init_method=dist_url,
        world_size=world_size,
        rank=rank,
        **kwargs,
    )
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)


def init_distributed_mode_slurm(backend="nccl", port=None, **kwargs):
    """Initialize slurm distributed training environment.

    If argument ``port`` is not specified, then the master port will be system
    environment variable ``MASTER_PORT``. If ``MASTER_PORT`` is not in system
    environment variable, then a default port ``29500`` will be used.

    Args:
        backend (str): Backend of torch.distributed.
        port (int, optional): Master port. Defaults to None.
    """
    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method("spawn")

    proc_id = int(os.environ["SLURM_PROCID"])
    ntasks = int(os.environ["SLURM_NTASKS"])
    node_list = os.environ["SLURM_NODELIST"]
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
    # specify master port
    if port is not None:
        os.environ["MASTER_PORT"] = str(port)
    elif "MASTER_PORT" in os.environ:
        pass  # use MASTER_PORT in the environment variable
    else:
        if _is_free_port(29500):
            os.environ['MASTER_PORT'] = '29500'
        else:
            os.environ['MASTER_PORT'] = str(_find_free_port())
    # use MASTER_ADDR in the environment variable if it already exists
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = addr
    os.environ["WORLD_SIZE"] = str(ntasks)
    os.environ["LOCAL_RANK"] = str(proc_id % num_gpus)
    os.environ["RANK"] = str(proc_id)

    world_size = ntasks
    gpu = proc_id % num_gpus
    rank = proc_id

    print(f"|| MASTER_ADDR={os.environ['MASTER_ADDR']} | MASTER_PORT={os.environ['MASTER_PORT']}")

    print(
        "|| dist_backend={} | world_size={} | global_rank={} | local_rank={}".format(
            backend,
            world_size,
            rank,
            gpu,
        ),
        flush=True,
    )

    dist.init_process_group(backend=backend)
    torch.distributed.barrier()
    if not DEBUG:
        setup_for_distributed(rank == 0)

    return gpu, rank, world_size


def setup_multi_processes(cfg):
    """Setup multi-processing environment variables."""

    # set multi-process start method
    if platform.system() != 'Windows':
        mp_start_method = cfg.get('mp_start_method', None)
        current_method = mp.get_start_method(allow_none=True)
        if mp_start_method in ('fork', 'spawn', 'forkserver'):
            print(
                f'|| Multi-processing start method `{mp_start_method}` is: '
                f'different from the previous setting `{current_method}`.'
                f'It will be force set to `{mp_start_method}`.')
            mp.set_start_method(mp_start_method, force=True)
        else:
            print(f'|| Multi-processing start method is:   `{mp_start_method}`')
            print(f'|| Multi-processing current method is: `{current_method}`')

    # disable opencv multithreading to avoid system being overloaded
    opencv_num_threads = cfg.get('opencv_num_threads', None)
    if isinstance(opencv_num_threads, int):
        print(f'|| OpenCV num_threads is: `{opencv_num_threads}`')
        cv2.setNumThreads(opencv_num_threads)
    else:
        print(f'|| OpenCV num_threads is: `{cv2.getNumThreads()}`')

    if cfg["num_workers"] > 1:
        # setup OMP threads
        # This code is referred from https://github.com/pytorch/pytorch/blob/master/torch/distributed/run.py  # noqa
        omp_num_threads = cfg.get('omp_num_threads', None)
        if 'OMP_NUM_THREADS' not in os.environ:
            if isinstance(omp_num_threads, int):
                print(f'|| OMP num threads is: {omp_num_threads}')
                os.environ['OMP_NUM_THREADS'] = str(omp_num_threads)
        else:
            print(f'|| OMP num threads is: {os.environ["OMP_NUM_THREADS"]}')

        # setup MKL threads
        if 'MKL_NUM_THREADS' not in os.environ:
            mkl_num_threads = cfg.get('mkl_num_threads', None)
            if isinstance(mkl_num_threads, int):
                print(f'|| MKL num threads is {mkl_num_threads}')
                os.environ['MKL_NUM_THREADS'] = str(mkl_num_threads)
        else:
            print(f'|| MKL num threads is: {os.environ["MKL_NUM_THREADS"]}')


# new functions from timm 2022-10-26
def is_distributed_env():
    if 'WORLD_SIZE' in os.environ:
        return int(os.environ['WORLD_SIZE']) >= 1  # only '>' to avoid DDP on single GPU
    if 'SLURM_NTASKS' in os.environ:
        return int(os.environ['SLURM_NTASKS']) >= 1  # only '>' to avoid DDP on single GPU
    return False


def world_info_from_env():
    local_rank = 0
    for v in ('LOCAL_RANK', 'MPI_LOCALRANKID', 'SLURM_LOCALID', 'OMPI_COMM_WORLD_LOCAL_RANK'):
        if v in os.environ:
            local_rank = int(os.environ[v])
            break

    global_rank = 0
    for v in ('RANK', 'PMI_RANK', 'SLURM_PROCID', 'OMPI_COMM_WORLD_RANK'):
        if v in os.environ:
            global_rank = int(os.environ[v])
            break

    world_size = 1
    for v in ('WORLD_SIZE', 'PMI_SIZE', 'SLURM_NTASKS', 'OMPI_COMM_WORLD_SIZE'):
        if v in os.environ:
            world_size = int(os.environ[v])
            break

    return local_rank, global_rank, world_size


def init_distributed_device(args):
    # Distributed training = training on more than one GPU.
    # Works in both single and multi-node scenarios.
    args.distributed = False
    args.world_size = 1
    args.rank = 0  # global rank
    args.local_rank = 0

    dist_backend = getattr(args, 'dist_backend', 'nccl')
    dist_url = getattr(args, 'dist_url', 'env://')
    if is_distributed_env():

        node_list = os.environ["SLURM_NODELIST"]
        addr = subprocess.getoutput(f"scontrol show hostname {node_list} | head -n1")
        # specify master port
        if "MASTER_PORT" in os.environ:
            pass  # use MASTER_PORT in the environment variable
        else:
            if _is_free_port(29500):
                os.environ['MASTER_PORT'] = '29500'
            else:
                os.environ['MASTER_PORT'] = str(_find_free_port())
        # use MASTER_ADDR in the environment variable if it already exists
        if "MASTER_ADDR" not in os.environ:
            os.environ["MASTER_ADDR"] = addr

        if 'SLURM_PROCID' in os.environ:
            # DDP via SLURM
            args.local_rank, args.rank, args.world_size = world_info_from_env()
            # SLURM var -> torch.distributed vars in case needed
            os.environ['LOCAL_RANK'] = str(args.local_rank)
            os.environ['RANK'] = str(args.rank)
            os.environ['WORLD_SIZE'] = str(args.world_size)
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
                world_size=args.world_size,
                rank=args.rank,
                timeout=datetime.timedelta(minutes=45)
            )
        else:
            # DDP via torchrun, torch.distributed.launch
            args.local_rank, _, _ = world_info_from_env()
            torch.distributed.init_process_group(
                backend=dist_backend,
                init_method=dist_url,
            )
            args.world_size = torch.distributed.get_world_size()
            args.rank = torch.distributed.get_rank()
        args.distributed = True

        print(f"|| MASTER_ADDR={os.environ['MASTER_ADDR']} | MASTER_PORT={os.environ['MASTER_PORT']}")

        print(
            "|| dist_backend={} | world_size={} | global_rank={} | local_rank={}".format(
                dist_backend,
                args.world_size,
                args.rank,
                args.local_rank,
            ),
            flush=True,
        )

    if torch.cuda.is_available():
        if args.distributed:
            device = 'cuda:%d' % args.local_rank
        else:
            device = 'cuda:0'
        torch.cuda.set_device(device)
    else:
        device = 'cpu'

    args.device = device
    args.device = torch.device(device)
