import torch
from parallel_ops.tensor_operations import partition_tensor
from parallel_ops.tensor_operations import combine_tensors
from parallel_ops.tensor_operations import replicate_tensor
from parallel_ops.tensor_operations import reduce_tensors
import logging

class Partition(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_mapping, dim):
        ctx.machine_mapping = machine_mapping
        ctx.dim = dim

        chunks = partition_tensor(tensor, machine_mapping, dim)
        ctx.chunks = chunks
        return chunks

    @staticmethod
    def backward(ctx, *grad_outputs):
        machine_mapping = ctx.machine_mapping
        dim = ctx.dim

        tensor = combine_tensors(grad_outputs, machine_mapping, dim)
        return tensor, None, None


class Replicate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor, machine_mapping):
        ctx.machine_mapping = machine_mapping

        tensors = replicate_tensor(tensor, machine_mapping)
        return tensors

    @staticmethod
    def backward(ctx, *grad_outputs):
        machine_mapping = ctx.machine_mapping

        full_grad = reduce_tensors(grad_outputs, machine_mapping)

        return full_grad, None

class Reduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensors, machine_mapping):
        logging.info(f"Reduce forward tensors{tensors}")
        ctx.machine_mapping = machine_mapping
        ctx.input_tensors = tensors
        
        tensor = reduce_tensors(tensors, machine_mapping)

        return tensor
        

    @staticmethod
    def backward(ctx, grad_output):
        logging.info("In reduce backward")
        machine_mapping = ctx.machine_mapping

        tensors = replicate_tensor(grad_output, machine_mapping)
        return tensors, None

