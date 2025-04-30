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
        return Combine.apply(input_values_all[self.parents[0]], dist.new_group(self.machine_view), self.dim)
    

class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_view, dim):
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

        if global_rank != 0:
            res = torch.empty((1), dtype=torch.float32).cuda(local_rank)
        
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

        return part_tensor, None, None