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
        m_view = dist.new_group(self.machine_view)
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()
        world_size = dist.get_world_size(m_view)

        if self.parents[0] in input_values_all:
            tensor = input_values_all[self.parents[0]]
        else:
            tensor = None
        
        chunks = None
        if global_rank == 0:
            chunks = list(torch.chunk(tensor, world_size, dim=self.dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
            
            ndim_tensor = torch.tensor([len(chunks[0].shape)], dtype=torch.long).cuda(local_rank)
        else:
            ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(ndim_tensor, src=0, group = m_view, async_op=False)
        ndim = ndim_tensor.item()

        if global_rank == 0:
            shape_tensor = torch.tensor(chunks[0].shape, dtype=torch.long).cuda(local_rank)
        else:
            shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(shape_tensor, src=0, group=m_view, async_op=False)
        shape = tuple(shape_tensor.tolist()) 

        part_tensor = torch.empty(shape, dtype=torch.float32, requires_grad=True).cuda(local_rank)
        
        return Partition.apply(part_tensor, m_view, self.dim, chunks)


class Partition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, part_tensor, machine_view, dim, chunks):
        ctx.machine_view = machine_view
        ctx.dim = dim
        
        dist.scatter(part_tensor, scatter_list=chunks, src=0, group=machine_view, async_op=False)
    
        return part_tensor
    
    @staticmethod
    def backward(ctx, grads):
        print(f"PART{grads}")
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
                
        return res, None, None, None