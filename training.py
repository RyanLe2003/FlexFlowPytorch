from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
import torch
from PCGNode import execute_node

def execute_pcg_parallel(pcg):
    results = {}
    num_gpus = torch.cuda.device_count()
    executor = ThreadPoolExecutor(max_workers=num_gpus)

    remaining_dependencies = {}
    for name, node in pcg.items():
        remaining_dependencies[name] = len(node.dependencies)

    def resolve_dependencies(node_name):
        for child in [n for n in pcg if node_name in pcg[n].dependencies]:
            remaining_dependencies[child] -= 1
            if remaining_dependencies[child] == 0:
                pcg[child].status = "READY"

    while any(node.status != "COMPLETED" for node in pcg.values()):
        ready_nodes = [name for name, node in pcg.items() if node.status == "READY"]

        futures = {}
        for node_name in ready_nodes:
            node = pcg[node_name]
            node.status = "RUNNING"
            node.outputs = [None] * len(node.machine_mapping)  # Initialize outputs for each task
            for index in range(len(node.machine_mapping)):
                future = executor.submit(execute_node, node, results, index)
                futures[future] = (node_name, index)
    
    for future in futures:
        node_name, task_idx = futures[future]
        results[node_name] = results.get(node_name, {})
        results[node_name][task_idx] = future.result()  # Store result for the specific task
        pcg[node_name].outputs[task_idx] = results[node_name][task_idx]  # Update node output

        # Check if all tasks for the node are completed
        if all(output is not None for output in pcg[node_name].outputs):
            pcg[node_name].status = "COMPLETED"
            resolve_dependencies(node_name)





    
    
        




    