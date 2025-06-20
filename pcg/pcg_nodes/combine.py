import torch
from pcg.pcg_nodes.pcg_node import PCGNode
import torch.distributed as dist
import os
from pcg.util.device_group_cache import device_group_cache
from pcg.pcg_nodes.parallel_tensor_attrs import *

class CombineNode(PCGNode):
    def __init__(
            self, 
            name: int, 
            parents: list, 
            parallel_tensor_attrs: ParallelTensorAttrs,
            machine_view: list, 
            dim: int):
        super().__init__(
            name=name, 
            parents=parents,
            parallel_tensor_attrs=parallel_tensor_attrs)
        self.machine_view = machine_view
        self.dim = dim
    
    def forward(self, name_to_node: map):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        # print(f"{global_rank}-{self.name}: Combine Start (Forward)")

        parent = name_to_node[self.parents[0]]

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return
        
        dev_group = device_group_cache(parent.machine_view)
        input = parent.data
        dst = self.machine_view[0]

        self.data = Combine.apply(input, dev_group, self.dim, dst, self.name)
        
        # print(f"{global_rank}-{self.name}: Combine Done (Forward)")


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            tensor: torch.Tensor, 
            dev_group: dist.ProcessGroup, 
            dim: int, 
            dst: int, 
            name: str):
        ctx.dev_group = dev_group
        ctx.dim = dim
        ctx.shape = tensor.shape
        ctx.dst = dst
        ctx.name = name

        world_size = dist.get_world_size(dev_group)
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        commun_proc = None
        machine_view = dist.get_process_group_ranks(dev_group)
        if dist in machine_view:
            commun_proc = dst
        else:
            commun_proc = machine_view[0]
        
        if (global_rank == dst):
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        else:
            gathered = None

        dist.gather(
            tensor=tensor, 
            gather_list=gathered, 
            dst=commun_proc, 
            group=dev_group, 
            async_op=False)

        if (global_rank == commun_proc):
            res = torch.cat(gathered, dim)
        else:
            output_shape = list(tensor.shape)
            output_shape[dim] *= world_size
            res = torch.empty(
                size=output_shape, 
                dtype=torch.float32).cuda(local_rank)  # dummy tensor
            # res = torch.empty((1), dtype=torch.float32).cuda(local_rank)
        
        return res
    
    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size(ctx.dev_group)

        print(f"{global_rank}-{ctx.name}: Combine Start (Backward): {grads}")

        chunks = None
        if global_rank == ctx.dst:
            chunks = list(torch.chunk(grads, world_size, dim=ctx.dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
        
        part_tensor = torch.empty(
            size=ctx.shape, 
            dtype=torch.float32).cuda(local_rank)

        dist.scatter(
            tensor=part_tensor, 
            scatter_list=chunks, 
            src=ctx.dst, 
            group=ctx.dev_group, 
            async_op=False)

        print(f"{global_rank}-{ctx.name}: Combine Done (Backward): {part_tensor}")

        return part_tensor, None, None, None, None
    