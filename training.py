from PCGNode import PCGNode
import torch

def partition_tensor(tensor, dim, num_partitions):
    return torch.chunk(tensor, num_partitions, dim)

def combine_tensors(tensors, dim):
    return torch.cat(tensors, dim)

def replicate_tensor(tensor, num_replicas):
    return [tensor.clone() for _ in range(num_replicas)]

def reduce_tensors(tensors):
    return torch.stack(tensors).sum(0)

def traverse(PCG):
    # topological sort
    indegree = {}
    for node_info in PCG.values():
        for child in node_info:
            indegree[child] = indegree.get(child, 0) + 1
    
    queue = []
    for node in indegree:
        if indegree[node] == 0:
            queue.append(node)
    
    while queue:
        cur_node = queue.pop()

        for child in PCG[cur_node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    
    

    



    








    




    
    






