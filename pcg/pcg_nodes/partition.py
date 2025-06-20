from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os
from pcg.util.device_group_cache import device_group_cache
from pcg.pcg_nodes.parallel_tensor_attrs import *

class PartitionNode(PCGNode):
    def __init__(
            self, 
            name: int, 
            parents: list, 
            parallel_tensor_attrs: ParallelTensorAttrs, 
            machine_view: list, 
            dim: int
            ):
        super().__init__(
            name=name, 
            parents=parents, 
            parallel_tensor_attrs=parallel_tensor_attrs)
        self.machine_view = machine_view
        self.dim = dim

    def forward(self, name_to_node: map):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        parent = name_to_node[self.parents[0]]

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return
        
        dev_group = device_group_cache(self.machine_view)
        input = parent.data

        output = torch.empty(
            self.get_shape(), 
            dtype=torch.float32, 
            requires_grad=True).cuda(local_rank)

        src = parent.machine_view[0]

        self.data = Partition.apply(input, output, self.dim, dev_group, src)


class Partition(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            input: torch.Tensor, 
            output: torch.Tensor, 
            dim: int, 
            dev_group: dist.ProcessGroup, 
            src: int
            ):
        ctx.dim = dim
        ctx.dev_group = dev_group
        ctx.src = src

        chunks = None
        if input is not None:
            chunks = list(torch.chunk(
                input=input, 
                chunks=dist.get_world_size(ctx.dev_group), 
                dim=dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
            
        dist.scatter(
            tensor=output, 
            scatter_list=chunks, 
            src=src, 
            group=dev_group, 
            async_op=False)

        return output
    
    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()
        world_size = dist.get_world_size(ctx.dev_group)

        print(f"{global_rank}: Partition Start (Backward): {grads}")
        
        if global_rank == ctx.src:
            gathered = [torch.empty_like(grads) for _ in range(world_size)]
        else:
            gathered = None
        
        dist.gather(
            tensor=grads, 
            gather_list=gathered, 
            dst=ctx.src, 
            group=ctx.dev_group, 
            async_op=False)

        if global_rank == ctx.src:
            res = torch.cat(tensors=gathered, dim=ctx.dim)
        else:
            res = None

        print(f"{global_rank}: Partition Done (Backward): {res}")
          
        return res, None, None, None, None