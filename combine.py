import torch
from pcg_node import PCGNode
import torch.distributed as dist
import os

class CombineNode(PCGNode):
    def __init__(self, name, parents, machine_view, dim):
        super().__init__(name, parents)
        self.machine_view = machine_view
        self.dim = dim
    
    def forward(self, input_values_all):
        return Combine.apply(dist.new_group(self.machine_view), self.dim, input_values_all[self.parents[0]])
    

class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, machine_view, dim, tensor):
        ctx.machine_view = machine_view
        ctx.dim = dim
        ctx.shape = tensor.shape
        world_size = dist.get_world_size(machine_view)

        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if global_rank == 0:
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        else:
            gathered = None

        dist.gather(tensor, gathered, 0, group=machine_view, async_op=False)  # commun proc at 0

        if global_rank == 0:
            res = torch.cat(gathered, dim)
            ndim_tensor = torch.tensor([len(res.shape)], dtype=torch.long).cuda(local_rank)
        else:
            ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)

        dist.broadcast(ndim_tensor, src=0, group=machine_view, async_op=False)
        ndim = ndim_tensor.item()

        if global_rank == 0:
            shape_tensor = torch.tensor(res.shape, dtype=torch.long).cuda(local_rank)
        else:
            shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)
        
        
        dist.broadcast(shape_tensor, src=0, group=machine_view, async_op=False)
        shape = tuple(shape_tensor.tolist())

        if global_rank != 0:
            res = torch.empty(shape, dtype=torch.float32).cuda(local_rank)
        
        return res
    
    @staticmethod
    def backward(ctx, grads):
        print(f"COMB{grads}")
        world_size = dist.get_world_size(ctx.machine_view)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        chunks = None
        if global_rank == 0:
            chunks = list(torch.chunk(grads, world_size, dim=ctx.dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
        
        part_tensor = torch.empty(ctx.shape, dtype=torch.float32, requires_grad=True).cuda(local_rank)

        dist.scatter(part_tensor, scatter_list=chunks, src=0, group=ctx.machine_view, async_op=False)  # commun proc at 0

        return None, None, part_tensor