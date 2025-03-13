from concurrent.futures import ThreadPoolExecutor
from node_status import node_status
from node_types import node_types
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor
from torch.distributed._tensor import Shard, Replicate, distribute_tensor
import torch
import torch.distributed as dist

def execute_pcg(pcg, local_target, num_iterations):
    remaining_parents = {}

    parameters = []
    parameters_node_order = []
    for id, node in pcg.items():

        if node.type == node_types.WEIGHT:
            parameters.append(node.input[0])
            parameters_node_order.append(node)

    learning_rate = 0.01
    optimizer = torch.optim.SGD(params=parameters, lr=learning_rate)
    
    for epoch in range(num_iterations):
        for id, node in pcg.items():
            node.status = node_status.WAITING if node.parents else node_status.READY
            remaining_parents[id] = len(node.parents)
            if node.type != node_types.INPUT and node.type != node_types.WEIGHT:
                node.input = []
    
        optimizer.zero_grad()

        # forward pass
        with ThreadPoolExecutor() as executor:
            while any(node.status != node_status.COMPLETED for node in pcg.values()):
                ready_nodes = [node for node in pcg.values() if node.status == node_status.READY]

                futures = []
                for node in ready_nodes:
                    futures.append(executor.submit(process_node, node, pcg, remaining_parents))
                
                for future in futures:
                    future.result()

        # compute loss

        # going to use replicate, replicate here temporarily
        # but think will have to change to a more general way 
        # to get the target to all valid machines.
        target_dt = DTensor.from_local(local_target, init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp")), [Replicate(), Replicate()])

        output_node = None
        for id, node in pcg.items():
            if node.type == node_types.OUTPUT:
                output_node = node
        
        loss_dt = ((output_node.output - target_dt) ** 2).mean()
        loss = loss_dt.to_local()

        # backward pass
        loss.backward(retain_graph=True)

        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")
        



        # print(f"Epoch {epoch}, Loss: {loss.item()}")
        # if dist.get_rank() == 0:
        #     print(f"PRINTING OUT PCG, EPOCH{epoch}")
        #     print_pcg(pcg)





def process_node(node, pcg, remaining_parents):
    node.status = node_status.RUNNING
    node.forward()
    node.status = node_status.COMPLETED

    for child in [n for n in pcg if node.id in pcg[n].parents]:
        remaining_parents[child] -= 1
        if remaining_parents[child] == 0:
            for parent in pcg[child].parents:
                # print(f"Appending to {pcg[child].id}: {pcg[parent].output}")
                pcg[child].input.append(pcg[parent].output)
            
            # print(f"After all appends: {pcg[child].input}")
            
            pcg[child].status = node_status.READY
    
            
# this is just for testing purposes
def print_pcg(pcg):
    for id, node in pcg.items():
        if node.type == node_types.WEIGHT:
            print(f"--------------------------")
            print(f"Node id: {node.id}")
            print(f"Node input: {node.input}")
            print(f"Node output: {node.output}")
            print(f"Node status: {node.status}")
    print(f"--------------------------")