from pcg.pcg_nodes.partition import PartitionNode
from pcg.pcg_nodes.replicate import ReplicateNode
from pcg.pcg_nodes.combine import CombineNode
from pcg.pcg_nodes.input import InputNode
from pcg.pcg_nodes.weight import WeightNode
from pcg.pcg_nodes.matmul import MatmulNode
from pcg.pcg_nodes.output import OutputNode

import torch.distributed as dist
import torch
import os

import pcg.util.topo_sort as ts
import pcg.pcg_train.train as train
import torch.nn as nn

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()

# manual pcg setup
weight_node = WeightNode(2, [], [0], (2, 2))
part_node = PartitionNode(3, [1], [0, 1], 0)
rep_node = ReplicateNode(4, [2], [0, 1])
matmul = MatmulNode(5, [3, 4], [0, 1])
combine = CombineNode(6, [5], [0], 0)
output = OutputNode(7, [6], [0])  # should be the same as weight_node

name_to_node = {
    2: weight_node,
    3: part_node,
    4: rep_node,
    5: matmul,
    6: combine,
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

params = None
if global_rank == 0:
    params = [weight_node.data[0]]
output_node = output

num_epochs = 1
target = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32).cuda(local_rank)
input_data = torch.tensor([[2, 3], [6, 7]], dtype=torch.float32, device=f'cuda:{local_rank}', requires_grad=True)
for i in range(num_epochs):
    input_node = InputNode(1, [], [0], input_data)
    name_to_node[1] = input_node
    train.train(
        order, 
        name_to_node, 
        target, 
        params, 
        output_node,
        loss_fn=nn.MSELoss(),
        optimizer = torch.optim.SGD(params, lr=0.01) if params else None
    )

print(f"AFTER TRAINING: {params}")

    








