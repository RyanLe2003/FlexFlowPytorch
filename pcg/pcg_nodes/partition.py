from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os
import pcg.util.move_tensor as mt

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
        
        new_data = []
        for src in parent.machine_view:
            for tensor in parent.data:
                dev_group = dist.new_group(self.machine_view)

                commun_proc = src
                if src not in self.machine_view:
                    shape_dev_group = dist.new_group([src, self.machine_view[0]])
                    if (global_rank == self.machine_view[0] or global_rank == src):
                        shape = mt.get_shape(src, tensor, shape_dev_group)

                        if (global_rank == src):
                            dist.send(tensor, dst=self.machine_view[0], group=shape_dev_group, async_op=False)
                            tensor = torch.empty((1), dtype=torch.float32).cuda(local_rank)
                        elif (global_rank == self.machine_view[0]):
                            tensor = torch.empty(shape, dtype=torch.float32).cuda(local_rank)
                            dist.recv(tensor, src=src, group=shape_dev_group, async_op=False)
        
                    commun_proc = self.machine_view[0]
                
                chunks = None
                if (global_rank == commun_proc):
                    tensor_cop = tensor.clone()
                    chunks = list(torch.chunk(tensor_cop, len(self.machine_view), dim=self.dim))
                    for i in range(len(chunks)):
                        chunks[i] = chunks[i].contiguous()
                
                if chunks is None:
                    shape = mt.get_shape(commun_proc, None, dev_group) 
                else:
                    shape = mt.get_shape(commun_proc, chunks[0], dev_group)

                part_tensor = torch.empty(shape, dtype=torch.float32, requires_grad=True).cuda(local_rank)
                part_tensor = Partition.apply(tensor, dev_group, self.dim, commun_proc, part_tensor, chunks)
                
                new_data.append(part_tensor)
                
        self.data = new_data


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