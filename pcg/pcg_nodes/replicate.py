from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

from pcg.util.device_group_cache import device_group_cache
from pcg.pcg_nodes.parallel_tensor_attrs import *

class ReplicateNode(PCGNode):
    def __init__(
            self, 
            name: int, 
            parents: list, 
            parallel_tensor_attrs: ParallelTensorAttrs,
            machine_view: list):
        super().__init__(
            name=name, 
            parents=parents,
            parallel_tensor_attrs=parallel_tensor_attrs)
        self.machine_view = machine_view
    
    def forward(self, name_to_node: map):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        # print(f"{global_rank}-{self.name}: Replicate Start (Forward)")

        parent = name_to_node[self.parents[0]]

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return
    
        dev_group = device_group_cache(self.machine_view)
        input = parent.data

        if input is None:
            input = torch.empty(
                size=self.get_shape(), 
                dtype=torch.float32, 
                requires_grad=True).cuda(local_rank)

        src = parent.machine_view[0]

        self.data = Replicate.apply(input, dev_group, src, self.name)

        # print(f"{global_rank}-{self.name}: Replicate Done (Forward)")


class Replicate(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            tensor: torch.Tensor, 
            device_group: dist.ProcessGroup, 
            src_rank: int, 
            name: str):
        ctx.device_group = device_group
        ctx.src_rank = src_rank
        ctx.name = name
        
        dist.broadcast(
            tensor=tensor, 
            src=src_rank, 
            group=device_group, 
            async_op=False)

        return tensor
    
    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()
        print(f"{global_rank}-{ctx.name}: Replicate Start (Backward): {grads}")

        grad_input = grads.clone()
        dist.reduce(
            tensor=grad_input, 
            dst=ctx.src_rank, 
            group=ctx.device_group, 
            async_op=False)

        print(f"{global_rank}-{ctx.name}: Replicate Done (Backward)")

        if global_rank == ctx.src_rank:
            return grad_input, None, None, None
        else:
            return None, None, None, None
        