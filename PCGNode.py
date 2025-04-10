import torch
import logging

from torch.autograd import Function
import torch.utils._pytree as pytree

logging.basicConfig(level=logging.DEBUG)
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

def pytreeify(cls):
    assert issubclass(cls, Function)

    orig_fw = cls.forward
    orig_bw = cls.backward
    orig_apply = cls.apply

    def new_apply(*inp):
        flat_inp, struct = pytree.tree_flatten(inp)
        out_struct_holder = []
        flat_out = orig_apply(struct, out_struct_holder, *flat_inp)
        assert len(out_struct_holder) == 1
        return pytree.tree_unflatten(flat_out, out_struct_holder[0])

    def new_forward(ctx, struct, out_struct_holder, *flat_inp):
        inp = pytree.tree_unflatten(flat_inp, struct)
        out = orig_fw(ctx, *inp)
        flat_out, out_struct = pytree.tree_flatten(out)
        ctx._inp_struct = struct
        ctx._out_struct = out_struct
        out_struct_holder.append(out_struct)
        return tuple(flat_out)

    def new_backward(ctx, *flat_grad_outputs):
        grad_outputs = pytree.tree_unflatten(flat_grad_outputs, ctx._out_struct)
        if not isinstance(grad_outputs, tuple):
            grad_outputs = (grad_outputs,)
        grad_inputs = orig_bw(ctx, *grad_outputs)
        flat_grad_inputs, grad_inputs_struct = pytree.tree_flatten(grad_inputs)
        if grad_inputs_struct != ctx._inp_struct:
            print(grad_inputs_struct)
            print(ctx._inp_struct)
            raise RuntimeError("The backward generated an arg structure that doesn't "
                               "match the forward's input.")
        return (None, None) + tuple(flat_grad_inputs)

    cls.apply = new_apply
    cls.forward = new_forward
    cls.backward = new_backward
    return cls

@pytreeify
class Combine(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensors, machine_mapping, dim):
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
        
        return grads,[None], None

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

@pytreeify
class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensors, machine_mapping):
        org_devices = []
        for tensor in tensors:
            org_devices.append(tensor.device)

        ctx.org_devices = org_devices

        res = 0
        for tensor in tensors: 
            res += tensor.to(machine_mapping[0])
        
        return res

    @staticmethod
    def backward(ctx, grad):
        clones = [grad.clone() for _ in range((len(ctx.org_devices)))]
        res = []

        for machine, clone in zip(ctx.org_devices, clones):
            clone = clone.to(machine)
            res.append(clone)
        
        return res, [None]

    