from pcg.pcg_nodes.partition import PartitionNode
from pcg.pcg_nodes.reduce import ReduceNode
from pcg.pcg_nodes.input import InputNode
from pcg.pcg_nodes.weight import WeightNode
from pcg.pcg_nodes.matmul import MatmulNode
from pcg.pcg_nodes.output import OutputNode

import torch.distributed as dist
import torch
import os

import torch.nn as nn

import pcg.util.topo_sort as ts

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()

# json processing (done manually rn)
input_data = [[2, 3], [6, 7]]
weight_data = [[1, 2], [3, 4]]

input_node = InputNode(1, [], [0], input_data)
weight_node = WeightNode(2, [], [0], weight_data)
part_node_1 = PartitionNode(3, [1], [0, 1], 1)
part_node_2 = PartitionNode(4, [2], [0, 1], 0)
matmul = MatmulNode(5, [3, 4], [0, 1])
reduce = ReduceNode(6, [5], [0])
output = OutputNode(7, [6], [0])  # should be the same as weight_node

name_to_node = {
    1: input_node,
    2: weight_node,
    3: part_node_1,
    4: part_node_2,
    5: matmul,
    6: reduce,
    7: output
}

graph = {
    1: [3],
    2: [4],
    3: [5],
    4: [5],
    5: [6],
    6: [7],
    7: [],
}

# get lexicographical topological sort
order = ts.get_order(graph)
print(f"exec order: {order}")
# order = [input_node, weight_node, part_node_1, part_node_2, matmul, reduce, output]

# train
def train(order, name_to_node, target):
    # forward pass
    for node in order:
        node.forward(name_to_node)
        print(f"{global_rank}: {node.data}")
    
    prediction = name_to_node[7].data
    if prediction is None:
        return
    
    # backward pass
    
    loss_fn = nn.MSELoss()
    loss = loss_fn(prediction, target)

    loss.backward()

    for node in order:
        print(f"{global_rank}: Name: {node.name} Leaf: {node.data.is_leaf}, requires_grad: {node.data.requires_grad}, grad: {node.data.grad}")


    # assuming weight device and output the same for now
    if global_rank in name_to_node[7].machine_view:
        optimizer = torch.optim.SGD([name_to_node[2].data], lr=0.01)  # need a clean way to get this
        optimizer.step()
        optimizer.zero_grad()

        res = name_to_node[2].data
        print(f"RESULT: {res}")



num_epochs = 1
target = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32).cuda(local_rank)
for i in range(num_epochs):
    train(order, name_to_node, target)

    








