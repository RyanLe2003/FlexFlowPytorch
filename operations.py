import torch

def partition_tensor(tensor, dim, num_partitions):
    return torch.chunk(tensor, num_partitions, dim)

def combine_tensors(tensors, dim):
    return torch.cat(tensors, dim)

def replicate_tensor(tensor, num_replicas):
    return [tensor.clone() for _ in range(num_replicas)]

def reduce_tensors(tensors):
    return torch.stack(tensors).sum(0)

    


    



    








    




    
    






