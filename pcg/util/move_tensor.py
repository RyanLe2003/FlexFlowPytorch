import torch.distributed as dist
import os
import torch

def get_shape(src, tensor, shape_dev_group):
    global_rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))

    if (global_rank == src):
        ndim_tensor = torch.tensor([len(tensor.shape)], dtype=torch.long).cuda(local_rank)
    else:
        ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)

    dist.broadcast(ndim_tensor, src=src, group=shape_dev_group, async_op=False)
    ndim = ndim_tensor.item()

    if (global_rank == src):
        shape_tensor = torch.tensor(tensor.shape, dtype=torch.long).cuda(local_rank)
    else:
        shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)

    dist.broadcast(shape_tensor, src=src, group=shape_dev_group, async_op=False)
    shape = tuple(shape_tensor.tolist())
    
    return shape
    
