import torch
import logging

logging.basicConfig(level=logging.DEBUG)

def partition_tensor(tensor, machine_mapping, dim):
    chunks = torch.chunk(tensor, len(machine_mapping), dim)

    for machine_id, chunk in zip(machine_mapping, chunks):
        logging.info(f"Sending tensor {chunk} to machine {machine_id}")
        device = torch.device(f"cuda:{machine_id}")
        chunk = chunk.to(device)
    
    return chunks

def combine_tensors(tensors, machine_mapping, dim):
    device = torch.device(f"cuda:{machine_mapping[0]}")
    for tensor in tensors:
        tensor = tensor.to(device)
    
    return torch.cat(tensors, dim)

def replicate_tensor(tensor, machine_mapping):
    clones = [tensor.clone() for _ in range(len(machine_mapping))]
    for machine_id, clone in zip(machine_mapping, clones):
        device = torch.device(f"cuda:{machine_id}")
        clone = clone.to(device)
    
    return clones

def reduce_tensors(tensors, machine_mapping):
    device = torch.device(f"cuda:{machine_mapping[0]}")
    for tensor in tensors:
        tensor = tensor.to(device)
    
    return torch.stack(tensors).sum(0)
