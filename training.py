from concurrent.futures import ThreadPoolExecutor
from parallel_ops.tensor_operations import partition_tensor
from parallel_ops.tensor_operations import combine_tensors
from parallel_ops.tensor_operations import replicate_tensor
from parallel_ops.tensor_operations import reduce_tensors
from parallel_ops.parallel_ops import operation_names
from node_status import node_status
import torch
import logging

logging.basicConfig(level=logging.DEBUG)


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
    logging.debug(f"Processing node: {node.name}, status: {node.status}")
    node.status = node_status.RUNNING
    parent_outputs = []
    for parent in node.dependencies:
        parent_node = pcg[parent]
        for data in parent_node.data:
            parent_outputs.append(data)
    
    logging.debug(f"{node.name} - Parent Data: {parent_outputs}")
    
    # send entire parent_outputs for combine, reduce, replication, partition
    if len(node.machine_mapping) == 1:
        logging.debug(f"{node.name} - Parallel op, MACHINE {node.machine_mapping[0]}")

        result = execute_node(node, parent_outputs, 0)
        
        node.data.append(result)
        logging.debug(f"{node.name} - Result: {node.data}")

    else:
        for index in range(len(node.machine_mapping)):
            logging.debug(f"{node.name} - Algebraic op, MACHINE {node.machine_mapping[index]}")

            result = executor.submit(execute_node, node, parent_outputs[index], index)

            node.append(result)
    
    node.status = node_status.COMPLETED
    logging.debug(f"{node.name} - Status: {node.status}")

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
        # for i in range(len(input)):
        #     input[i] = input[i].to(torch_device)
        
        if node.operation == operation_names.PARTITION:
            logging.debug(f"Partitioning tensor {input[0]}")
            return partition_tensor(input[0], node.dim, node.num_partitions)
        elif node.operation == operation_names.COMBINE:
            logging.debug(f"Combining tensors {input}")
            return combine_tensors(input[0], node.dim)
        elif node.operation == operation_names.REPLICATE:
            logging.debug(f"Replicating tensor {input[0]}")
            return replicate_tensor(input[0], node.num_replicas)
        elif node.operation == operation_names.REDUCE:
            logging.debug(f"Reducing tensors {input}")
            return reduce_tensors(input[0])
    else:
        return None
    