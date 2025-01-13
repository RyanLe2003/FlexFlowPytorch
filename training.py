from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import torch
from PCGNode import execute_node

def execute_pcg_parallel(pcg):
    num_gpus = torch.cuda.device_count()
    executor = ThreadPoolExecutor(max_workers=num_gpus)

    remaining_dependencies = {}
    for name, node in pcg.items():
        remaining_dependencies[name] = len(node.dependencies)

    while any(node.status != "COMPLETED" for node in pcg.values()):
        ready_nodes = [node for node in pcg.values() if node.status == "READY"]
        for node in ready_nodes:
            node.status = "RUNNING"
            
            parent_outputs = []
            for parent in node.dependencies:
                for parent_node in pcg[parent]:
                    for data in parent_node.data:
                        parent_outputs.append(data)
            
            # assuming len(parent_oupts) == len(node.machine_mapping)
            for index in range(len(node.machine_mapping)):
                result = executor.submit(execute_node, node, parent_outputs, index)
                node.data.append(result)
            
            node.status = "COMPLETED"
            for child in [n for n in pcg if node.name in pcg[n].dependencies]:
                remaining_dependencies[child] -= 1
                if remaining_dependencies[child] == 0:
                    pcg[child].status = "READY"
    