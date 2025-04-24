from pcg_node import PCGNode
import torch
import torch.distributed as dist

class ReduceNode(PCGNode):
    def __init__(self, name, parents, machine_view):
        super().__init__(name, parents)
        self.machine_view = machine_view

    def forward(self, input_values_all):
        return Reduce.apply(dist.new_group(self.machine_view), input_values_all[self.parent[0]])

class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, machine_view, tensor):
        ctx.machine_view = machine_view

        tensor_cop = tensor.clone()
        dist.reduce(tensor_cop, dst=0, group=machine_view, async_op=False)

        global_rank = dist.get_rank()

        if global_rank == 0:
            return tensor_cop
        else:
            return None

    @staticmethod
    def backward(ctx, grads):
        dist.broadcast(grads, src=0, group=ctx.machine_view, async_op=False)  # commun proc at 0
        return grads, None