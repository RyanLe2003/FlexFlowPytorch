from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

class PartitionNode(PCGNode):
    def __init__(self, name, parents, machine_view, dim):
        super().__init__(name, parents)
        self.machine_view = machine_view
        self.dim = dim

    def forward(self, name_to_node):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        # determine which processes are relevant
        parent = name_to_node[self.parents[0]]  # partition only has one parent

        if (global_rank not in parent.machine_view and
            global_rank not in self.machine_view):
            return
        
        # for src in parent.machine_view (ASSUMING ALL PARENTS ARE A SINGLE TENSOR)
        chunks = None
        src = parent.machine_view[0]
        dev_group = dist.new_group(self.machine_view)
        
        if (global_rank == src):
            tensor = parent.data
            tensor_cop = tensor.clone()
            chunks = list(torch.chunk(tensor_cop, len(self.machine_view), dim=self.dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
        
        # broadcast number of dimensions in a chunk
        if (global_rank == src):
            ndim_tensor = torch.tensor([len(chunks[0].shape)], dtype=torch.long).cuda(local_rank)
        else:
            ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(ndim_tensor, src=src, group=dev_group, async_op=False)
        ndim = ndim_tensor.item()

        # broadcast dimensions of a chunk
        if global_rank == src:
            shape_tensor = torch.tensor(chunks[0].shape, dtype=torch.long).cuda(local_rank)
        else:
            shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)
        
        dist.broadcast(shape_tensor, src=src, group=dev_group, async_op=False)
        shape = tuple(shape_tensor.tolist())

        # create partitions
        part_tensor = torch.empty(shape, dtype=torch.float32, requires_grad=True).cuda(local_rank)

        part_tensor = Partition.apply(parent.data, dev_group, self.dim, src, part_tensor, chunks)
        
        self.data = part_tensor


class Partition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, device_group, dim, src_rank, part_tensor, chunks):
        ctx.device_group = device_group
        ctx.dim = dim
        ctx.src_rank = src_rank

        dist.scatter(part_tensor, scatter_list=chunks, src=src_rank, group=device_group, async_op=False)
    
        return part_tensor
    
    @staticmethod
    def backward(ctx, grads):
        global_rank = dist.get_rank()
        world_size = dist.get_world_size(ctx.device_group)

        print(f"PARTITION, {global_rank}: {grads}")
        
        if global_rank == ctx.src_rank:
            gathered = [torch.empty_like(grads) for _ in range(world_size)]
        else:
            gathered = None
        
        # this includes the dummy tensor -> could cause issues if None
        dist.gather(grads, gathered, ctx.src_rank, group=ctx.device_group, async_op=False)

        if global_rank == ctx.src_rank:
            res = torch.cat(gathered, ctx.dim)
        else:
            res = None

        print(f"PARTITION AFTER, {global_rank}: {res}")
          
        return res, None, None, None, None, None