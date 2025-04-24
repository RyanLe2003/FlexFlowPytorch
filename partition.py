from pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

class PartitionNode(PCGNode):
    def __init__(self, name, parents, machine_view, dim):
        super().__init__(name, parents)
        self.machine_view = machine_view
        self.dim = dim

    def forward(self, input_values_all):
        if self.parents[0] in input_values_all:
            tensor = input_values_all[self.parents[0]]
        else:
            tensor = None
        
        return Partition.apply(tensor, dist.new_group(self.machine_view), self.dim)


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
        
        dist.scatter(part_tensor, scatter_list=chunks, src=0, group=machine_view, async_op=False)

        return part_tensor
    
    @staticmethod
    def backward(ctx, grads):
        world_size = dist.get_world_size(ctx.machine_view)
        global_rank = dist.get_rank()

        if global_rank == 0:
            gathered = [torch.empty_like(grads) for _ in range(world_size)]
        else:
            gathered = None
        
        dist.gather(grads, gathered, 0, group=ctx.machine_view, async_op=False)

        if global_rank == 0:
            res = torch.cat(gathered, ctx.dim)
        else:
            res = None
                
        return res, None, None