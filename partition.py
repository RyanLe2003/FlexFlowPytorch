from pcg_node import PCGNode
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
        
        if (global_rank == src):
            tensor = parent.data
            tensor_cop = tensor.clone()
            chunks = list(torch.chunk(tensor_cop, len(self.machine_view), dim=self.dim))
            for i in range(len(chunks)):
                chunks[i] = chunks[i].contiguous()
            
        # include src device if not part of dst device
        mod_m_view = self.machine_view
        is_dest_incl = True
        if src not in mod_m_view:
            mod_m_view.append(src)
            is_dest_incl = False

        dev_group = dist.new_group(mod_m_view)
        
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

        # add dummy tensor for src
        if global_rank == src and not is_dest_incl:
            chunks.append(part_tensor.clone())

        part_tensor = Partition.apply(part_tensor, dev_group, self.dim, src, chunks)
        if global_rank in self.machine_view:
            self.data = part_tensor
        


        # m_view = dist.new_group(self.machine_view)

        # local_rank = int(os.environ.get("LOCAL_RANK", 0))
        # global_rank = dist.get_rank()
        # world_size = dist.get_world_size(m_view)

        # if self.parents[0] in input_values_all:
        #     tensor = input_values_all[self.parents[0]]
        #     is_commun_proc = torch.tensor(global_rank, dtype=torch.int32).cuda(local_rank)
        # else:
        #     tensor = None
        #     is_commun_proc = torch.tensor(0, dtype=torch.int32).cuda(local_rank)
        
        # # get global rank of tensor before op
        # src_rank_tensor = torch.zeros(1, dtype=torch.int32)
        # dist.all_reduce(is_commun_proc, op=dist.ReduceOp.SUM, group=m_view)
        # src_rank = src_rank_tensor.item()
   
        # chunks = None
        # if global_rank == src_rank:
        #     chunks = list(torch.chunk(tensor, world_size, dim=self.dim))
        #     for i in range(len(chunks)):
        #         chunks[i] = chunks[i].contiguous()
            
        #     ndim_tensor = torch.tensor([len(chunks[0].shape)], dtype=torch.long).cuda(local_rank)
        # else:
        #     ndim_tensor = torch.empty(1, dtype=torch.long).cuda(local_rank)
        
        # dist.broadcast(ndim_tensor, src=src_rank, group = m_view, async_op=False)
        # ndim = ndim_tensor.item()

        # if global_rank == src_rank:
        #     shape_tensor = torch.tensor(chunks[0].shape, dtype=torch.long).cuda(local_rank)
        # else:
        #     shape_tensor = torch.empty(ndim, dtype=torch.long).cuda(local_rank)
        
        # dist.broadcast(shape_tensor, src=src_rank, group=m_view, async_op=False)
        # shape = tuple(shape_tensor.tolist()) 

        # part_tensor = torch.empty(shape, dtype=torch.float32, requires_grad=True).cuda(local_rank)
        
        # return Partition.apply(part_tensor, m_view, self.dim, chunks, src_rank)


class Partition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, part_tensor, device_group, dim, src_rank, chunks):
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
          
        return res, None, None, None, None