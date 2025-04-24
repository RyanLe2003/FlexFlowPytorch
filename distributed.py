import torch.distributed as dist
import torch
import os

from torch.distributed.device_mesh import DeviceMesh
from torch.distributed._tensor import DTensor, distribute_tensor, Shard, Replicate, Partial

class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_view):
        ctx.machine_view = machine_view

        tensor_cop = tensor.clone()
        dist.reduce(tensor_cop, dst=0, group=machine_view, async_op=False)  # commun proc at 0

        global_rank = dist.get_rank()

        if global_rank == 0:
            return tensor_cop
        else:
            return None
    
    @staticmethod
    def backward(ctx, grads):
        dist.broadcast(grads, src=0, group=ctx.machine_view, async_op=False)  # commun proc at 0
        return grads


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_view, dim):
        ctx.machine_view = machine_view
        ctx.dim = dim
        world_size = dist.get_world_size(machine_view)

        global_rank = dist.get_rank()

        if global_rank == 0:
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        else:
            gathered = None

        dist.gather(part_tensor, gathered, 0, group=m_view_2, async_op=False)  # commun proc at 0

        if global_rank == 0:
            res = torch.cat(gathered, dim)
        else:
            res = None
        
        return res
    
    @staticmethod
    def backward(ctx, grads):
        world_size = dist.get_world_size(ctx.machine_view)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        chunks = None
        if global_rank == 0:
            chunks = list(torch.chunk(grads, world_size, dim=ctx.dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
        
        part_tensor = torch.empty((2, 1), dtype=torch.float32).cuda(local_rank) # need to figure out how to dynamically get shape
        dist.scatter(part_tensor, scatter_list=chunks, src=0, group=ctx.machine_view, async_op=False)  # commun proc at 0

        return part_tensor, None, None


class Replicate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_view):
        ctx.machine_view = machine_view
        dist.broadcast(tensor, src=0, group=machine_view, async_op=False)  # commun proc at 0
        return tensor
    
    @staticmethod
    def backward(ctx, grads):
        grad_input = grads.clone() #reduce is in place
        dist.reduce(grad_input, dst=0, group=ctx.machine_view, async_op=False)  # commun proc at 0

        if dist.get_rank() == 0:
            return grad_input, None
        else:
            # return torch.zeros_like(grad_input), None
            return None
    
class Partition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_view, dim):
        ctx.machine_view = machine_view
        ctx.dim = dim
        world_size = dist.get_world_size(machine_view)
        
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        chunks = None
        if global_rank == 0:
            chunks = list(torch.chunk(tensor, world_size, dim=dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
            
            ndim_tensor = torch.tensor([len(chunks[0].shape)], dtype=torch.long).cuda(local_rank)
        else:
            ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(ndim_tensor, src=0, group = machine_view, async_op=False)
        ndim = ndim_tensor.item()

        if global_rank == 0:
            shape_tensor = torch.tensor(chunks[0].shape, dtype=torch.long).cuda(local_rank)
        else:
            shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(shape_tensor, src=0, group=machine_view, async_op=False)
        shape = tuple(shape_tensor.tolist()) 

        part_tensor = torch.empty(shape, dtype=torch.float32).cuda(local_rank)
        
        
        dist.scatter(part_tensor, scatter_list=chunks, src=0, group=machine_view, async_op=False)  # commun proc at 0

        return part_tensor
    
    @staticmethod
    def backward(ctx, grads):
        world_size = dist.get_world_size(ctx.machine_view)
        global_rank = dist.get_rank()

        if global_rank == 0:
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        else:
            gathered = None
        
        dist.gather(grads, gathered, 0, group=ctx.machine_view, async_op=False)  # commun proc at 0

        if global_rank == 0:
            res = torch.cat(gathered, ctx.dim)
        else:
            res = None
                
        return res, None, None

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()

# node 1 (Replicate)
ranks_m_view_1 = [0, 1]
m_view_1 = dist.new_group(ranks_m_view_1)
tensor = None
if global_rank in ranks_m_view_1:
    if global_rank == 0:  # communication process
        tensor = torch.tensor([[2, 3], [6, 7]], dtype=torch.float32).cuda(local_rank)
        ndim_tensor = torch.tensor([len(tensor.shape)], dtype=torch.long).cuda(local_rank)
    else:
        ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)
    
    dist.broadcast(ndim_tensor, src=0, group=m_view_1, async_op=False)
    ndim = ndim_tensor.item()

    if global_rank == 0:
        shape_tensor = torch.tensor(tensor.shape, dtype=torch.long).cuda(local_rank)
    else:
        shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)
    
    dist.broadcast(shape_tensor, src=0, group=m_view_1, async_op=False)
    shape = tuple(shape_tensor.tolist())   

    if global_rank != 0:
        tensor = torch.empty(shape, dtype=torch.float32).cuda(local_rank)

    tensor = Replicate.apply(tensor, m_view_1)

print(tensor)

# node 2 (Partition)
ranks_m_view_2 = [0, 1]
m_view_2 = dist.new_group(ranks_m_view_2)
part_tensor = None
if global_rank in ranks_m_view_2:
    if global_rank == 0:
        weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).cuda(local_rank)
    else:
        weight = None

    part_tensor = Partition.apply(weight, m_view_2, 1)

print(part_tensor)
    
# node 3 (combine)
ranks_m_view_3 = [0, 1]
m_view_3 = dist.new_group(ranks_m_view_3)
combine_tensor = None
if global_rank in ranks_m_view_3:
    combine_tensor = Combine.apply(part_tensor, m_view_3, 1)

print(combine_tensor)


# node 4 (reduce)
ranks_m_view_4 = [0, 1]
m_view_4 = dist.new_group(ranks_m_view_4)
reduce_tensor = None
if global_rank in ranks_m_view_4:
    reduce_tensor = Reduce.apply(tensor, m_view_4)

print(reduce_tensor)

dist.destroy_process_group()










