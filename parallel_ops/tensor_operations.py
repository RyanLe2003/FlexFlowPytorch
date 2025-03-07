import torch
import logging

logging.basicConfig(level=logging.DEBUG)

def partition_tensor(tensor, machine_mapping, dim):
    # logging.info(f"checking if tensor requires grad: {tensor.requires_grad}")
    chunks = torch.chunk(tensor, len(machine_mapping), dim)
    res = []

    for machine_id, chunk in zip(machine_mapping, chunks):
        device = torch.device(f"cuda:{machine_id}")
        chunk = chunk.to(device)
        chunk = chunk.requires_grad_(tensor.requires_grad)
        # logging.info(f"checking if chunk requires grad: {chunk.requires_grad}")
        res.append(chunk)
    
    return res

def combine_tensors(tensors, machine_mapping, dim):
    device = torch.device(f"cuda:{machine_mapping[0]}")
    for tensor in tensors:
        tensor = tensor.to(device)
    
    return [torch.cat(tensors, dim)]

def replicate_tensor(tensor, machine_mapping):
    clones = [tensor.clone() for _ in range(len(machine_mapping))]
    res = []
    for machine_id, clone in zip(machine_mapping, clones):
        device = torch.device(f"cuda:{machine_id}")
        clone = clone.to(device)
        clone = clone.requires_grad_(tensor.requires_grad)
        res.append(clone)
    
    
    return res

def reduce_tensors(tensors, machine_mapping):
    device = torch.device(f"cuda:{machine_mapping[0]}")
    res = []
    for tensor in tensors:
        temp = tensor.to(device)
        temp = temp.requires_grad_(tensor.requires_grad)
        res.append(temp)

    
    logging.info(f"res before reduce: {res}")
    reduced_tensor = torch.sum(torch.stack(res), dim=0)
    logging.info(f"reduced tensor: {reduced_tensor}")

    reduced_tensor = reduced_tensor.requires_grad_()
    
    return [reduced_tensor]
