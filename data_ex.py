from partition import PartitionNode
from replicate import ReplicateNode
from combine import CombineNode
from input import InputNode
from weight import WeightNode
from matmul import MatmulNode
from output import OutputNode

import torch.distributed as dist
import torch
import os

import torch.nn as nn

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()

# this will manually do the json processing step (nodes + train order)
input_data = [[2, 3], [6, 7]]
weight_data = [[1, 2], [3, 4]]

input_node = InputNode("input", [], [0], input_data)
weight_node = WeightNode("weight", [], [0], weight_data)
part_node = PartitionNode("input_part", ["input"], [0, 1], 0)
rep_node = ReplicateNode("weight_rep", ["weight"], [0, 1])
matmul = MatmulNode("matmul", ["input_part", "weight_rep"], [0, 1])
combine = CombineNode("combine", ["matmul"], [0], 0)
output = OutputNode("output", ["combine"], [0])  # should be the same as weight_node

name_to_node = {
    "input": input_node,
    "weight": weight_node,
    "input_part": part_node,
    "weight_rep": rep_node,
    "matmul": matmul,
    "combine": combine,
    "output": output
}

order = [input_node, weight_node, part_node, rep_node, matmul, combine, output]

# train
def train(order, name_to_node, target):
    # forward pass
    for node in order:
        node.forward(name_to_node)
        print(f"{global_rank}: {node.data}")
    
    prediction = name_to_node["output"].data
    if prediction is None:
        return
    
    # backward pass
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(prediction, target)

    loss.backward()

    # assuming weight device and output the same for now
    if global_rank in name_to_node["output"].machine_view:
        optimizer = torch.optim.SGD([name_to_node["weight"].data], lr=0.01)  # need a clean way to get this
        optimizer.step()
        optimizer.zero_grad()

        res = name_to_node["weight"].data
        print(f"RESULT: {res}")


num_epochs = 5
target = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32).cuda(local_rank)
for i in range(num_epochs):
    train(order, name_to_node, target)

    








