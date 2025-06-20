from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

from pcg.util.device_group_cache import device_group_cache
from pcg.pcg_nodes.parallel_tensor_attrs import *

class ReduceNode(PCGNode):
    def __init__(
            self, 
            name: int, 
            parents: list, 
            parallel_tensor_attrs: ParallelTensorAttrs, 
            machine_view: list
            ):
        super().__init__(
            name=name, 
            parents=parents, 
            parallel_tensor_attrs=parallel_tensor_attrs)
        self.machine_view = machine_view

    def forward(self, name_to_node: map):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        # print(f"{global_rank}-{self.name}: Reduce Start (Forward)")

        parent = name_to_node[self.parents[0]]

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return
        
        device_group = device_group_cache(parent.machine_view)
        input = parent.data
        dst = self.machine_view[0]

        self.data = Reduce.apply(input, device_group, dst, self.name)

        # print(f"{global_rank}-{self.name}: Reduce Done (Forward)")


class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(
            ctx, 
            tensor: torch.Tensor, 
            device_group: dist.ProcessGroup, 
            dst: int, 
            name: str
            ):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ctx.device_group = device_group
        ctx.dst = dst
        ctx.name = name

        tensor_cop = tensor.clone()
        dist.reduce(tensor_cop, dst=dst, group=device_group, async_op=False)

        if (global_rank == dst):
            res = tensor_cop
        else:
            res = torch.empty(
                tensor_cop.shape, 
                dtype=torch.float32).cuda(local_rank)
        
        return res

    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()
        print(f"{global_rank}-{ctx.name}: Reduce Start (Backward): {grads}")

        dist.broadcast(
            tensor=grads, 
            src=ctx.dst, 
            group=ctx.device_group, 
            async_op=False)

        print(f"{global_rank}-{ctx.name}: Reduce Done (Backward): {grads}")
        
        return grads, None, None, None
    