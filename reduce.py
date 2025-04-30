from pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

class ReduceNode(PCGNode):
    def __init__(self, name, parents, machine_view):
        super().__init__(name, parents)
        self.machine_view = machine_view

    def forward(self, name_to_node):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        parent = name_to_node[self.parents[0]]

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return

        # going to assume here dst U src = src
        device_group = dist.new_group(parent.machine_view)
        tensor = parent.data
        dst = self.machine_view[0]

        self.data = Reduce.apply(tensor, device_group, dst)


class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, device_group, dst):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        ctx.device_group = device_group
        ctx.dst = dst

        tensor_cop = tensor.clone()
        dist.reduce(tensor_cop, dst=dst, group=device_group, async_op=False)

        if global_rank == dst:
            return tensor_cop
        else:
            return torch.empty(tensor_cop.shape, dtype=torch.float32).cuda(local_rank)

    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()
        print(f"REDUCE, {global_rank}: {grads}")
        dist.broadcast(grads, src=ctx.dst, group=ctx.device_group, async_op=False)

        print(f"REDUCE AFTER, {global_rank}: {grads}")
        
        return grads, None, None