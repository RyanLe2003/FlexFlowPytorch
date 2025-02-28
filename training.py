from concurrent.futures import ThreadPoolExecutor
from node_status import node_status
import logging
import threading
import os
from node_types import node_types

logging.basicConfig(level=logging.DEBUG)


def execute_pcg(pcg):
    logging.info("Executing PCG")
    cpu_count = os.cpu_count()
    print(f"Available CPU cores: {cpu_count}")

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
        
        # logging.info("Executing PCG - Backward Pass")
        backward_pass(pcg, executor)


def backward_pass(pcg, executor):
    remaining_children = {}
    for id, node in pcg.items():
        remaining_children[id] = len([n for n in pcg.values() if id in n.parents])

    while any(node.backward_status != node_status.COMPLETED for node in pcg.values()):
        ready_nodes = [node for node in pcg.values() if node.backward_status == node_status.READY]


        if not ready_nodes:
            logging.warning("No ready nodes in backward pass. Possible deadlock.")
            break
        
        futures = []
        for node in ready_nodes:
            futures.append(executor.submit(process_node_backward, node, pcg, remaining_children))
        
        for future in futures:
            future.result()

def process_node(node, pcg, remaining_parents):
    logging.info(f"Processing node: {node.id}, status: {node.status}, thread: {threading.get_ident()}")

    if node.type == node_types.OUTPUT:
        logging.info(f"Output node: {node.id}, data: {node.data}")

    node.status = node_status.RUNNING
    node.forward_pass()
    node.status = node_status.COMPLETED

    logging.info(f"Completed node: {node.id}, status: {node.status}")

    for child in [n for n in pcg if node.id in pcg[n].parents]:
        remaining_parents[child] -= 1
        if remaining_parents[child] == 0:
            for parent in pcg[child].parents:
                pcg[child].data.append(pcg[parent].data)

            pcg[child].status = node_status.READY


def process_node_backward(node, pcg, remaining_children):
    logging.info(f"Processing node backward: {node.id}, status: {node.backward_status}, tensor grad: {node.grad}")

    node.backward_status = node_status.RUNNING
    node.backward_pass(pcg)
    node.backward_status = node_status.COMPLETED

    logging.info(f"Completed node backward: {node.id}, status: {node.backward_status}, tensor grad: {node.grad}")

    for parent_id in node.parents:
        remaining_children[parent_id] -= 1
        parent_node = pcg[parent_id]
        if remaining_children[parent_id] == 0:
            if parent_node.grad is None:
                parent_node.grad = node.grad
            else:
                parent_node.grad = [g1 + g2 for g1, g2 in zip(parent_node.grad, node.grad)]

            pcg[parent_id].backward_status = node_status.READY
    