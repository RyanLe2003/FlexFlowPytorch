import torch.distributed as dist
import torch
import os

def setup():
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(local_rank)
    
    torch.manual_seed(42)


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0

def is_parallel():
    return dist.is_available() and dist.is_initialized()

