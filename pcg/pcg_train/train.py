import torch
import torch.nn as nn

import torch.distributed as dist
from pcg.pcg_nodes.weight import WeightNode

def train(order, name_to_node, target, params, output_node, loss_fn, optimizer):
    global_rank = dist.get_rank()
    # forward pass
    for node in order:
        name_to_node[node].forward(name_to_node)

    if len(output_node.data) != 1:
        raise RuntimeError("incorrect number of results produced")
    
    prediction = output_node.data[0]
    # if prediction is None:
    #     return
    
    # backward pass
    loss = prediction
    
    loss = loss_fn(prediction, target)

    if global_rank in output_node.machine_view:
        print(f"LOSS {loss}")
    loss.backward()

    print(f"{global_rank}: done w backward")

    if params:
        optimizer.step()        
        optimizer.zero_grad()

        print(f"{global_rank}: done w optimizer")

    for name in order:
        node = name_to_node[name]
        if not isinstance(node, WeightNode):
            for tensor in node.data:
                if tensor is not None:
                    tensor.detach()
            node.data = []
    
    print(f"{global_rank}: done w empty data")
            
    del prediction, loss