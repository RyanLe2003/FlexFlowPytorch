from concurrent.futures import ThreadPoolExecutor
from node_status import node_status
import logging

logging.basicConfig(level=logging.DEBUG)


def execute_pcg(pcg):
    logging.debug("Executing PCG")
    remaining_parents = {}
    for id, node in pcg.items():
        remaining_parents[id] = len(node.parents)

    with ThreadPoolExecutor() as executor:
        while any(node.status != node_status.COMPLETED for node in pcg.values()):
            ready_nodes = [node for node in pcg.values() if node.status == node_status.READY]
            
            futures = []
            for node in ready_nodes:
                futures.append(executor.submit(process_node, node, pcg, remaining_parents))
            
            for future in futures:
                future.result()
        
        logging.debug("Executing PCG - Backward Pass")
        backward_pass(pcg, executor)


def backward_pass(pcg, executor):
    remaining_children = {}
    for id, node in pcg.items():
        remaining_children[id] = len([n for n in pcg.values() if id in n.parents])

    while any(node.backward_status != node_status.COMPLETED for node in pcg.values()):
        ready_nodes = [node for node in pcg.values() if node.backward_status == node_status.READY]
        
        futures = []
        for node in ready_nodes:
            futures.append(executor.submit(process_node_backward, node, pcg, remaining_children))
        
        for future in futures:
            future.result()


def process_node(node, pcg, remaining_parents):
    logging.debug(f"Processing node: {node.id}, status: {node.status}")

    node.status = node_status.RUNNING
    node.forward_pass()
    node.status = node_status.COMPLETED

    logging.debug(f"Completed node: {node.id}, status: {node.status}")

    for child in [n for n in pcg if node.id in pcg[n].parents]:
        remaining_parents[child] -= 1
        if remaining_parents[child] == 0:
            pcg[child].status = node_status.READY


def process_node_backward(node, pcg, remaining_children):
    logging.debug(f"Processing node backward: {node.id}, status: {node.backward_status}")

    node.backward_status = node_status.RUNNING
    node.backward_pass()
    node.backward_status = node_status.COMPLETED

    logging.debug(f"Completed node backward: {node.id}, status: {node.backward_status}")

    for parent_id in node.parents:
        remaining_children[parent_id] -= 1
        if remaining_children[parent_id] == 0:
            pcg[parent_id].backward_status = node_status.READY
    