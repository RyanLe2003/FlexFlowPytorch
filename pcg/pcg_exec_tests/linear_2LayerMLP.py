from torchvision import datasets, transforms
import torch

from pcg.pcg_nodes.input import InputNode
from pcg.pcg_nodes.weight import WeightNode
from pcg.pcg_nodes.matmul import MatmulNode
from pcg.pcg_nodes.relu import ReluNode
from pcg.pcg_nodes.output import OutputNode

import pcg.util.topo_sort as ts
import pcg.pcg_train.train as train
import torch.nn as nn

import torch.distributed as dist
import torch
import os
from pcg.pcg_nodes.parallel_tensor_attrs import *
from pcg.util.check_dist import *

setup()
local_rank = int(os.environ.get("LOCAL_RANK", 0))
global_rank = get_rank()

# manual pcg setup
weight_node_1_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(8, 1), ShardParallelDim(4, 1)], 
            ReplicaParallelDim(1, 1)
        )
    )
)
weight_1 = WeightNode(2, [], [0], weight_node_1_attrs)

weight_node_2_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(4, 1), ShardParallelDim(2, 1)], 
            ReplicaParallelDim(1, 1)
        )
    )
)
weight_2 = WeightNode(3, [], [0], weight_node_2_attrs)

matmul_1_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(1, 1), ShardParallelDim(4, 1)], 
            ReplicaParallelDim(1, 1)
        )
    )
)
matmul_1 = MatmulNode(4, [1, 2], [0], matmul_1_attrs)

relu_1_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(1, 1), ShardParallelDim(4, 1)], 
            ReplicaParallelDim(1, 1)
        )
    )
)
relu_1 = ReluNode(5, [4], [0], relu_1_attrs)

matmul_2_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(1, 1), ShardParallelDim(2, 1)], 
            ReplicaParallelDim(1, 1)
        )
    )
)
matmul_2 = MatmulNode(6, [5, 3], [0], matmul_2_attrs)

relu_2_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(1, 1), ShardParallelDim(2, 1)], 
            ReplicaParallelDim(1, 1)
        )
    )
)
relu_2 = ReluNode(7, [6], [0], relu_2_attrs)

output_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(1, 1), ShardParallelDim(2, 1)], 
            ReplicaParallelDim(1, 1)
        )
    )
)
output = OutputNode(8, [7], [0], output_attrs)

name_to_node = {
    2: weight_1,
    3: weight_2,
    4: matmul_1,
    5: relu_1,
    6: matmul_2,
    7: relu_2,
    8: output
}

graph = {
    1: [4],
    2: [4],
    3: [6],
    4: [5],
    5: [6],
    6: [7],
    7: [8],
    8: [],
}

# get lexicographical topological sort
order = ts.get_order(graph)

params = None
if global_rank == 0:
    params = [weight_1.data, weight_2.data]
output_node = output

epochs = 10
target = torch.randint(0, 2, size=(1,), device=f'cuda:{local_rank}')

input_data = torch.rand(
    size=(1, 8), 
    dtype=torch.float32, 
    device=f'cuda:{local_rank}', 
    requires_grad=True)
input_node_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(1, 1), ShardParallelDim(8, 1)], 
            ReplicaParallelDim(1, 1)
        )
    )
)
for i in range(epochs):
    input_node = InputNode(1, [], [0], input_data, input_node_attrs)
    name_to_node[1] = input_node

    train.train(
        order=order,
        name_to_node=name_to_node,
        target=target,
        params=params,
        output_node=output_node,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=torch.optim.SGD(params, lr=0.01) if params else None
    )

print("DONE")
    