from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

class ReplicateNode(PCGNode):
    def __init__(self, name, parents, machine_view):
        super().__init__(name, parents)
        self.machine_view = machine_view
    
    def forward(self, name_to_node):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        # determine which processes are relevant
        parent = name_to_node[self.parents[0]]  # replicate only has one parent

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return
        
        src = parent.machine_view[0]
        dev_group = dist.new_group(self.machine_view)

        if (global_rank == src):
            tensor = parent.data
            ndim_tensor = torch.tensor([len(tensor.shape)], dtype=torch.long).cuda(local_rank)
        else:
            ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(ndim_tensor, src=src, group=dev_group, async_op=False)
        ndim = ndim_tensor.item()

        if (global_rank == src):
            shape_tensor = torch.tensor(tensor.shape, dtype=torch.long).cuda(local_rank)
        else:
            shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(shape_tensor, src=src, group=dev_group, async_op=False)
        shape = tuple(shape_tensor.tolist())

        if global_rank != src:
            tensor = torch.empty(shape, device=f'cuda:{local_rank}', dtype=torch.float32, requires_grad=True)

        self.data = Replicate.apply(tensor, dev_group, src)


class Replicate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, device_group, src_rank):
        ctx.device_group = device_group
        ctx.src_rank = src_rank
        
        dist.broadcast(tensor, src=src_rank, group=device_group, async_op=False)

        return tensor
    
    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()

        grad_input = grads.clone()
        dist.reduce(grad_input, dst=ctx.src_rank, group=ctx.device_group, async_op=False)  # commun proc at 0

        if global_rank == ctx.src_rank:
            return grad_input, None, None
        else:
            return None, None, None