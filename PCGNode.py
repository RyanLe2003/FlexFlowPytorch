import torch

class PCGNode:
    def __init__(self, name, parents):
        self.name = name
        self.parents = parents
    
    def forward(self, input_values_all):
        raise NotImplementedError
    

class PartitionNode(PCGNode):
    def __init__(self, name, parents, machine_mapping, dim):
        super().__init__(name, parents)
        self.machine_mapping = machine_mapping
        self.dim = dim

    def forward(self, input_values_all):
        return Partition.apply(input_values_all[self.parents[0]], self.machine_mapping, self.dim)

class Partition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_mapping, dim):
        ctx.input_device = tensor.device
        ctx.dim = dim

        chunks = torch.chunk(tensor, len(machine_mapping), dim)
        res = []

        for machine, chunk in zip(machine_mapping, chunks):
            chunk = chunk.to(machine)
            res.append(chunk)
        
        return tuple(res)
    
    @staticmethod
    def backward(ctx, *grads):
        grads_on_input_device = [g.to(ctx.input_device) for g in grads]

        grad_input = torch.cat(grads_on_input_device, ctx.dim)
        return grad_input, None, None


    