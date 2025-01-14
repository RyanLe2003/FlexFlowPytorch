from concurrent.futures import ThreadPoolExecutor
from parallel_ops.tensor_operations import partition_tensor
from parallel_ops.tensor_operations import combine_tensors
from parallel_ops.tensor_operations import replicate_tensor
from parallel_ops.tensor_operations import reduce_tensors
from parallel_ops.operation_names import operation_names
from node_status import node_status
import torch

def execute_pcg(pcg):
    remaining_dependencies = {}
    for name, node in pcg.items():
        remaining_dependencies[name] = len(node.dependencies)

    with ThreadPoolExecutor() as executor:
        while any(node.status != node_status.COMPLETED for node in pcg.values()):
            ready_nodes = [node for node in pcg.values() if node.status == node_status.READY]
            
            futures = []
            for node in ready_nodes:
                futures.append(executor.submit(process_node, node, pcg, remaining_dependencies, executor))
            
            for future in futures:
                future.result()


def process_node(node, pcg, remaining_dependencies, executor):
    node.status = node_status.RUNNING
    parent_outputs = []
    for parent in node.dependencies:
        for parent_node in pcg[parent]:
            for data in parent_node.data:
                parent_outputs.append(data)
    
    # send entire parent_outputs for combine, reduce
    if len(node.machine_mapping) == 1:
        result = execute_node(node, parent_outputs, 0)
        node.data.append(result)
    else:
        for index in range(len(node.machine_mapping)):
            result = executor.submit(execute_node, node, parent_outputs[index], index)
            node.data.append(result)
    
    node.status = node_status.COMPLTED
    for child in [n for n in pcg if node.name in pcg[n].dependencies]:
        remaining_dependencies[child] -= 1
        if remaining_dependencies[child] == 0:
            pcg[child].status = node_status.READY


def execute_node(node, input, index):
    torch_device = torch.device("cpu")
    if torch.cuda.is_available():
        device = node.machine_mapping[index]
        torch_device = torch.device(f"cuda:{device}")
    
    if node.operation:
        for i in range(len(input)):
            input[i] = input[i].to(torch_device)
        
        if node.operation == operation_names.PARTITION:
            return partition_tensor(input[0], node.dim, node.num_partitions)
        elif node.operation == operation_names.COMBINE:
            return combine_tensors(input, node.dim)
        elif node.operation == operation_names.REPLICATE:
            return replicate_tensor(input[0], node.num_replicas)
        elif node.operation == operation_names.REDUCE:
            return reduce_tensors(input)
    else:
        return None
    