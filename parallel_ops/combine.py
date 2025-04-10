import torch
from pcg_node import PCGNode

class CombineNode(PCGNode):
    def __init__(self, name, parents, machine_mapping, dim):
        super().__init__(name, parents)
        self.machine_mapping = machine_mapping
        self.dim = dim
    
    def forward(self, input_values_all):
        return Combine.apply(self.machine_mapping, self.dim, input_values_all[self.parents[0]])
    

class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, machine_mapping, dim, *tensors):
        org_devices = []
        tensors_on_mm = []

        for tensor in tensors:
            org_devices.append(tensor.device)
            tensor = tensor.to(machine_mapping[0])
            tensors_on_mm.append(tensor)
          
        ctx.org_devices = org_devices
        ctx.dim = dim
        
        res = torch.cat(tensors_on_mm, dim)

        return res

    @staticmethod
    def backward(ctx, grad):
        chunks = torch.chunk(grad, len(ctx.org_devices), ctx.dim)
        grads = []
        for machine, chunk in zip(ctx.org_devices, chunks):
            chunk = chunk.to(machine)
            grads.append(chunk)
        
        return  None, None, *grads