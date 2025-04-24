from pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

class ReplicateNode(PCGNode):
    def __init__(self, name, parents, machine_view):
        super().__init__(name, parents)
        self.machine_view = machine_view
    
    def forward(self, input_values_all):
        if self.parents[0] in input_values_all:
            tensor = input_values_all[self.parents[0]]
        else:
            tensor = None

        return Replicate.apply(tensor, dist.new_group(self.machine_view))


class Replicate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_view):
        ctx.machine_view = machine_view

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        if global_rank == 0:  # communication process
            ndim_tensor = torch.tensor([len(tensor.shape)], dtype=torch.long).cuda(local_rank)
        else:
            ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(ndim_tensor, src=0, group=machine_view, async_op=False)
        ndim = ndim_tensor.item()

        if global_rank == 0:
            shape_tensor = torch.tensor(tensor.shape, dtype=torch.long).cuda(local_rank)
        else:
            shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(shape_tensor, src=0, group=machine_view, async_op=False)
        shape = tuple(shape_tensor.tolist())

        if global_rank != 0:
            tensor = torch.empty(shape, dtype=torch.float32).cuda(local_rank)
        
        dist.broadcast(tensor, src=0, group=machine_view, async_op=False)
        return tensor
    
    @staticmethod
    def backward(ctx, grads):
        grad_input = grads.clone() #reduce is in place
        dist.reduce(grad_input, dst=0, group=ctx.machine_view, async_op=False)  # commun proc at 0

        if dist.get_rank() == 0:
            return grad_input, None
        else:
            return None, None