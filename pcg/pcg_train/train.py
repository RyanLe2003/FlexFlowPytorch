import torch
import torch.nn as nn

import torch.distributed as dist
from pcg.pcg_nodes.weight import WeightNode
from pcg.util.check_dist import *

first_run = True

def train(order, name_to_node, target, params, output_node, loss_fn, optimizer):
    global first_run

    global_rank = get_rank()
    # forward pass
    for node in order:
        name_to_node[node].forward(name_to_node)
    prediction = output_node.data
    
    # backward pass
    loss = prediction
    
    loss = loss_fn(prediction, target)

    if global_rank in output_node.machine_view:
        print(f"LOSS {loss}")
    loss.backward()

    print(f"{global_rank}: done w backward")

    if params:
        optimizer.step()     
        print(f"{global_rank}: done w optimizer")

    for name in order:
        node = name_to_node[name]
        if not isinstance(node, WeightNode):
            node.data = None
    
    if params:
        optimizer.zero_grad()
    
    print(f"{global_rank}: done w empty data")

    mode = "w" if first_run else "a"
    first_run = False

    if is_parallel() and global_rank in output_node.machine_view:
        with open("logs/result_parallel.txt", mode) as f:
            for param in params:
                f.write(f"{param}")
    elif not is_parallel():
        with open("logs/result_nonparallel.txt", mode) as f:
            for param in params:
                f.write(f"{param}")

    del prediction, loss
    