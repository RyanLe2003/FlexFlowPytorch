import torch.nn as nn

from combine import CombineNode
from partition import PartitionNode
from replicate import ReplicateNode
from reduce import ReduceNode
from matmul import MatmulNode
from output import OutputNode

def parse_graph(json_graph):
    """
    Assumptions: 
    - json graph is topologically sorted
    - Reduce, Combine may have parallelism going in and will have len(M.M) == 1
    """
    node_map = {}
    dependency_graph = {}

    input = None
    input_name = None

    weights = nn.ParameterDict()

    refactor_info = {}

    for node in json_graph:
        node_type = node["type"]
        node_name = node["name"]
        node_parents = node.get("parents", [])

        dependency_graph_parents = []
        for parent in node_parents:
            if parent != input_name and parent not in weights.keys():
                dependency_graph_parents.append(parent)

        if node_type == "Input":
            input = node["value"]
            input_name = node_name
        
        if node_type == "Weight":
            weights[node_name] = nn.Parameter(node["value"])
            
        if node_type == "Output":
            obj = OutputNode("output", node_parents)
            node_map["output"] = obj

            dependency_graph["output"] = dependency_graph_parents

        if node_type == "Matmul":
            for i in range(len(node["machine_mapping"])):
                name_refac = f"{node_name}_{i}"
                par_refac = [f"{par}_{i}" for par in node_parents]            
                obj = MatmulNode(name_refac, par_refac)

                node_map[name_refac] = obj
                dependency_graph[name_refac] = dependency_graph_parents
            
            refactor_info[node_name] = [f"{node_name}_{i}" for i in range(len(node["machine_mapping"]))]
            
        if node_type == "Partition":
            obj = PartitionNode(node_name, node_parents, node["machine_mapping"], node["dim"])

            node_map[node_name] = obj
            dependency_graph[node_name] = dependency_graph_parents

            refactor_info[node_name] = [f"{node_name}_{i}" for i in range(len(node["machine_mapping"]))]
        
        if node_type == "Replicate":
            obj = ReplicateNode(node_name, node_parents, node["machine_mapping"])

            node_map[node_name] = obj
            dependency_graph[node_name] = dependency_graph_parents

            refactor_info[node_name] = [f"{node_name}_{i}" for i in range(len(node["machine_mapping"]))]
        
        if node_type == "Reduce":
            if node_parents[0] in refactor_info:
                dependency_graph[node_name] = refactor_info[node_parents[0]]
                obj = ReduceNode(node_name, refactor_info[node_parents[0]], node["machine_mapping"])   
            else:
                dependency_graph[node_name] = dependency_graph_parents
                obj = ReduceNode(node_name, node_parents, node["machine_mapping"]) 

            node_map[node_name] = obj  
        
        if node_type == "Combine":
            obj = CombineNode(node_name, node_parents, node["machine_mapping"], node["dim"])

            node_map[node_name] = obj
            
            if node_parents[0] in refactor_info:
                dependency_graph[node_name] = refactor_info[node_parents[0]]
            else:
                dependency_graph[node_name] = dependency_graph_parents
    
    return input, weights, node_map, dependency_graph







        


        
















