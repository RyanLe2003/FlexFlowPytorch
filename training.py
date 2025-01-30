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

    