import torch.nn as nn

from parallel_ops.combine import CombineNode
from parallel_ops.partition import PartitionNode
from parallel_ops.replicate import ReplicateNode
from parallel_ops.reduce import ReduceNode
from parallel_ops.matmul import MatmulNode

def parse_graph(json_graph):
    """
    Assumptions: 
    - json graph is topologically sorted
    - Reduce, Combine may have parallelism going in and will have len(M.M) == 1
    """
    node_map = {}
    dependency_graph = {}

    input = None
    weights = nn.ParameterDict()

    refactor_info = {}

    for node in json_graph:
        node_type = node["type"]
        node_name = node["name"]
        node_parents = node.get("parents", [])

        if node_type == "Input":
            input = node["value"]
            continue
        
        if node_type == "Weight":
            weights[node_name] = nn.Parameter(node["value"])
            continue

        if node_type == "Matmul":
            for i in range(len(node["machine_mapping"])):
                name_refac = f"{node_name}_{i}"
                par_refac = [f"{par}_{i}" for par in node_parents]            
                obj = MatmulNode(name_refac, par_refac)

                node_map[name_refac] = obj
                dependency_graph[name_refac] = par_refac
            
            refactor_info[node_name] = [f"{node_name}_{i}" for i in range(len(node["machine_mapping"]))]
            
        if node_type == "Partition":
            obj = PartitionNode(node_name, node_parents, node["machine_mapping"], node["dim"])

            node_map[node_name] = obj
            dependency_graph[node_name] = node_parents

            refactor_info[node_name] = [f"{node_name}_{i}" for i in range(len(node["machine_mapping"]))]
        
        if node_type == "Replicate":
            obj = ReplicateNode(node_name, node_parents, node["machine_mapping"])

            node_map[node_name] = obj
            dependency_graph[node_name] = node_parents

            refactor_info[node_name] = [f"{node_name}_{i}" for i in range(len(node["machine_mapping"]))]
        
        if node_type == "Reduce":
            obj = ReduceNode(node_name, node_parents, node["machine_mapping"])       

            node_map[node_name] = obj

            if node_parents[0] in refactor_info:
                dependency_graph[node_name] = refactor_info[node_parents[0]]
            else:
                dependency_graph[node_name] = node_parents
        
        if node_type == "Combine":
            obj = CombineNode(node_name, node_parents, node["machine_mapping"], node["dim"])

            node_map[node_name] = obj
            
            if node_parents[0] in refactor_info:
                dependency_graph[node_name] = refactor_info[node_parents[0]]
            else:
                dependency_graph[node_name] = node_parents
    
    return input, weights, node_map, dependency_graph







        


        
















