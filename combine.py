import torch
from pcg_node import PCGNode
import torch.distributed as dist
import os

class CombineNode(PCGNode):
    def __init__(self, name, parents, machine_view, dim):
        super().__init__(name, parents)
        self.machine_view = machine_view
        self.dim = dim
    
    def forward(self, name_to_node):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        # determine which processes are relevant
        parent = name_to_node[self.parents[0]]

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return
        
        device_group = dist.new_group(parent.machine_view)
        tensor = parent.data
        dst = self.machine_view[0]
        
        self.data =  Combine.apply(tensor, device_group, self.dim, dst)
    

class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, device_group, dim, dst):
        ctx.device_group = device_group
        ctx.dim = dim
        ctx.shape = tensor.shape
        ctx.dst = dst
        world_size = dist.get_world_size(device_group)

        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        if (global_rank == dst):
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        else:
            gathered = None

        dist.gather(tensor, gathered, dst, group=device_group, async_op=False)

        if (global_rank == dst):
            res = torch.cat(gathered, dim)
        else:
            res = torch.empty((1), dtype=torch.float32).cuda(local_rank)  # dummy tensor
        
        return res
    
    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size(ctx.device_group)

        chunks = None
        if global_rank == ctx.dst:
            chunks = list(torch.chunk(grads, world_size, dim=ctx.dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
        
        part_tensor = torch.empty(ctx.shape, dtype=torch.float32, requires_grad=True).cuda(local_rank)

        dist.scatter(part_tensor, scatter_list=chunks, src=ctx.dst, group=ctx.device_group, async_op=False)  # commun proc at 0

        return part_tensor, None, None, None