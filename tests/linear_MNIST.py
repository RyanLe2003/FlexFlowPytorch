from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

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

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transforms.ToTensor()
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transforms.ToTensor()  
)

batch_size = 64
def build_graph(input_tensor, weight_1, weight_2):
    input_node_attrs = ParallelTensorAttrs(
        ParallelTensorShape(
            ParallelTensorDim(
                [ShardParallelDim(batch_size, 1), ShardParallelDim(784, 1)], 
                ReplicaParallelDim(1, 1)
            )
        )
    )
    input_node = InputNode(1, [], [0], input_tensor, input_node_attrs)
    
    matmul_1_attrs = ParallelTensorAttrs(
        ParallelTensorShape(
            ParallelTensorDim(
                [ShardParallelDim(batch_size, 1), ShardParallelDim(512, 1)], 
                ReplicaParallelDim(1, 1)
            )
        )
    )
    matmul_1 = MatmulNode(
        4, 
        [1, 2], 
        [0],
        matmul_1_attrs
        )

    relu_1_attrs = ParallelTensorAttrs(
        ParallelTensorShape(
            ParallelTensorDim(
                [ShardParallelDim(batch_size, 1), ShardParallelDim(512, 1)], 
                ReplicaParallelDim(1, 1)
            )
        )
    )
    relu_1 = ReluNode(5, [4], [0], relu_1_attrs)

    matmul_2_attrs = ParallelTensorAttrs(
        ParallelTensorShape(
            ParallelTensorDim(
                [ShardParallelDim(batch_size, 1), ShardParallelDim(10, 1)], 
                ReplicaParallelDim(1, 1)
            )
        )
    )
    matmul_2 = MatmulNode(6, [5, 3], [0], matmul_2_attrs)

    relu_2_attrs = ParallelTensorAttrs(
        ParallelTensorShape(
            ParallelTensorDim(
                [ShardParallelDim(batch_size, 1), ShardParallelDim(10, 1)], 
                ReplicaParallelDim(1, 1)
            )
        )
    )
    relu_2 = ReluNode(7, [6], [0], relu_2_attrs)

    output_node_attrs = ParallelTensorAttrs(
        ParallelTensorShape(
            ParallelTensorDim(
                [ShardParallelDim(batch_size, 1), ShardParallelDim(10, 1)], 
                ReplicaParallelDim(1, 1)
            )
        )
    )
    output = OutputNode(8, [7], [0], output_node_attrs)

    name_to_node = {
        1: input_node,
        2: weight_1,
        3: weight_2,
        4: matmul_1,
        5: relu_1,
        6: matmul_2,
        7: relu_2,
        8: output
    }

    return name_to_node, output

graph = {
    1: [4],
    2: [4],
    3: [6],
    4: [5],
    5: [6],
    6: [7],
    7: [8],
    8: []
}

weight_1_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(784, 1), ShardParallelDim(512, 1)],
            ReplicaParallelDim(1, 1)
        )
    )
)
weight_1 = WeightNode(2, [], [0], weight_1_attrs)

weight_2_attrs = ParallelTensorAttrs(
    ParallelTensorShape(
        ParallelTensorDim(
            [ShardParallelDim(512, 1), ShardParallelDim(10, 1)],
            ReplicaParallelDim(1, 1)
        )
    )
)
weight_2 = WeightNode(3, [], [0], weight_2_attrs)

# get lexicographical topological sort
order = ts.get_order(graph)

params = None
if global_rank == 0:  # temp
    params = [weight_1.data, weight_2.data]

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    drop_last=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=True
)
epochs = 2
for i in range(epochs):
    print(f"{global_rank}: STARTING EPOCH {i + 1}")
    j = 1
    for batch_images, batch_labels in train_loader:
        print(f"{global_rank}: STARTING BATCH {j}")
        # Reshape to [batch_size, 784] from [batch_size, 1, 28, 28]
        batch_images_flat = batch_images.reshape(
            batch_images.shape[0], -1).to(
                device=f'cuda:{local_rank}')
        batch_labels = batch_labels.to(device=f'cuda:{local_rank}')
        
        # current batch
        name_to_node, output_node = build_graph(
            batch_images_flat, 
            weight_1, 
            weight_2
            )

        train.train(
            order=order,
            name_to_node=name_to_node,
            target=batch_labels,
            params=params,
            output_node=output_node,
            loss_fn=nn.CrossEntropyLoss(),
            optimizer=torch.optim.SGD(params, lr=0.01) if params else None
        )

        print(f"{global_rank}: DONE W BATCH {j}")
        j += 1
    print(f"{global_rank}: DONE W EPOCH {i}")



print("DONE")