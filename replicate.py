from pcg_node import PCGNode
import torch

class ReplicateNode(PCGNode):
    def __init__(self, name, parents, machine_mapping):
        super().__init__(name, parents)
        self.machine_mapping = machine_mapping
    
    def forward(self, input_values_all):
        return Replicate.apply(input_values_all[self.parents[0]], self.machine_mapping)


class Replicate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_mapping):
        ctx.input_device = tensor.device

        clones = [tensor.clone() for _ in range((len(machine_mapping)))]
        res = []

        for machine, clone in zip(machine_mapping, clones):
            clone = clone.to(machine)
            res.append(clone)
        
        return tuple(res)
    
    @staticmethod
    def backward(ctx, *grads):
        grads_on_input_device = [g.to(ctx.input_device) for g in grads]
        
        res = 0
        for g in grads_on_input_device:
            res += g
        
        return res, None