from torchvision import datasets, transforms
import torch
from torch.utils.data import DataLoader

from pcg.pcg_nodes.partition import PartitionNode
from pcg.pcg_nodes.replicate import ReplicateNode
from pcg.pcg_nodes.combine import CombineNode
from pcg.pcg_nodes.reduce import ReduceNode
from pcg.pcg_nodes.input import InputNode
from pcg.pcg_nodes.weight import WeightNode
from pcg.pcg_nodes.matmul import MatmulNode
from pcg.pcg_nodes.relu import ReluNode
from pcg.pcg_nodes.softmax import SoftMaxNode
from pcg.pcg_nodes.output import OutputNode

import pcg.util.topo_sort as ts
import pcg.pcg_train.train as train
import torch.nn as nn

import torch.distributed as dist
import torch
import os
import gc

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()

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

def build_graph(input_tensor, weight_1, weight_2):
    input_node = InputNode(1, [], [0], input_tensor)
    input_partition = PartitionNode(4, [1], [0, 1], dim=1)
    weight_1_partition = PartitionNode(5, [2], [0, 1], dim=0)
    weight_2_partition = PartitionNode(6, [3], [0, 1], dim=1)
    matmul_1 = MatmulNode(7, [4, 5], [0, 1])
    reduce_1 = ReduceNode(8, [7], [0])
    replicate_1 = ReplicateNode(9, [8], [0, 1])
    relu_1 = ReluNode(10, [9], [0, 1])
    matmul_2 = MatmulNode(11, [10, 6], [0, 1])
    relu_2 = ReluNode(12, [11], [0, 1])
    combine_1 = CombineNode(13, [12], [0], dim=1)
    output = OutputNode(14, [13], [0])

    name_to_node = {
        1: input_node,
        2: weight_1,
        3: weight_2,
        4: input_partition,
        5: weight_1_partition,
        6: weight_2_partition,
        7: matmul_1,
        8: reduce_1,
        9: replicate_1,
        10: relu_1,
        11: matmul_2,
        12: relu_2,
        13: combine_1,
        14: output
    }

    return name_to_node, output

graph = {
    1: [4],
    2: [5],
    3: [6],
    4: [7],
    5: [7],
    6: [11],
    7: [8],
    8: [9],
    9: [10],
    10: [11],
    11: [12],
    12: [13],
    13: [14],
    14: []
}

weight_1 = WeightNode(2, [], [0], (784, 512))
weight_2 = WeightNode(3, [], [0], (512, 10))

# get lexicographical topological sort
order = ts.get_order(graph)

params = None
if global_rank == 0:  # temp
    params = [weight_1.data[0], weight_2.data[0]]

batch_size = 64
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False
)
epochs = 1
for i in range(epochs):
    print(f"{global_rank}: STARTING EPOCH {i + 1}")
    j = 1
    for batch_images, batch_labels in train_loader:
        print(f"{global_rank}: STARTING BATCH {j}")
        # Reshape to [batch_size, 784] from [batch_size, 1, 28, 28]
        batch_images_flat = batch_images.reshape(batch_images.shape[0], -1).to(device=f'cuda:{local_rank}')
        batch_labels = batch_labels.to(device=f'cuda:{local_rank}')
        
        # current batch
        name_to_node, output_node = build_graph(batch_images_flat, weight_1, weight_2)

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
    



    




