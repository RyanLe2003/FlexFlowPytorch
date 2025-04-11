from pcg_node import PCGNode
import torch

class ReduceNode(PCGNode):
    def __init__(self, name, parents, machine_mapping):
        super().__init__(name, parents)
        self.machine_mapping = machine_mapping

    def forward(self, input_values_all):
        tensors = []
        for parent in self.parents:
            tensors.append(input_values_all[parent])
        return Reduce.apply(self.machine_mapping, *tensors)

class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, machine_mapping, *tensors):
        org_devices = []
        for tensor in tensors:
            org_devices.append(tensor.device)

        ctx.org_devices = org_devices

        res = 0
        for tensor in tensors: 
            res += tensor.to(machine_mapping[0])
        
        return tuple([res])

    @staticmethod
    def backward(ctx, grad):
        clones = [grad.clone() for _ in range((len(ctx.org_devices)))]
        res = []

        for machine, clone in zip(ctx.org_devices, clones):
            clone = clone.to(machine)
            res.append(clone)
        
        return None, *res