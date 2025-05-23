import torch
from pcg.pcg_nodes.pcg_node import PCGNode
import torch.distributed as dist
import os
import pcg.util.move_tensor as mt
from pcg.util.device_group_cache import device_group_cache

class CombineNode(PCGNode):
    def __init__(self, name, parents, machine_view, dim):
        super().__init__(name, parents)
        self.machine_view = machine_view
        self.dim = dim
    
    def forward(self, name_to_node):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        # print(f"{global_rank}-{self.name}: Combine Start (Forward)")

        # determine which processes are relevant
        parent = name_to_node[self.parents[0]]

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return
        
        new_data = []

        # only support one for now
        assert len(parent.data) == 1, f"Expected one local tensor, got {len(parent.data)}"

        for tensor in parent.data:
            dst = self.machine_view[0]
            res = Combine.apply(tensor, parent.machine_view, self.dim, dst, self.name)
            new_data.append(res)

        self.data = new_data

        # print(f"{global_rank}-{self.name}: Combine Done (Forward)")


class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_view, dim, dst, name):
        ctx.machine_view = machine_view
        ctx.dim = dim
        ctx.shape = tensor.shape
        ctx.dst = dst
        ctx.name = name

        device_group = device_group_cache(machine_view)
        ctx.device_group = device_group
        world_size = dist.get_world_size(device_group)

        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        commun_proc = None
        if dst in machine_view:
            commun_proc = dst
        else:
            commun_proc = machine_view[0]
        
        if (global_rank == commun_proc):
            gathered = [torch.empty_like(tensor) for _ in range(world_size)]
        else:
            gathered = None

        dist.gather(tensor, gathered, commun_proc, group=device_group, async_op=False)

        if (global_rank == commun_proc):
            res = torch.cat(gathered, dim)
        else:
            output_shape = list(tensor.shape)
            output_shape[dim] *= world_size
            res = torch.empty(output_shape, dtype=torch.float32).cuda(local_rank)  # dummy tensor
            # res = torch.empty((1), dtype=torch.float32).cuda(local_rank)
        
        return res
    
    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        world_size = dist.get_world_size(ctx.device_group)

        print(f"{global_rank}-{ctx.name}: Combine Start (Backward): {grads}")

        chunks = None
        if global_rank == ctx.dst:
            chunks = list(torch.chunk(grads, world_size, dim=ctx.dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
        
        part_tensor = torch.empty(ctx.shape, dtype=torch.float32).cuda(local_rank)

        dist.scatter(part_tensor, scatter_list=chunks, src=ctx.dst, group=ctx.device_group, async_op=False)

        print(f"{global_rank}-{ctx.name}: Combine Done (Backward): {part_tensor}")

        return part_tensor, None, None, None, None