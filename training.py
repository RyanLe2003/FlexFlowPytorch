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

    