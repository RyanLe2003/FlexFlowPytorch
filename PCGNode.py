from node_status import node_status
from parallel_ops.tensor_operations import partition_tensor
from parallel_ops.tensor_operations import combine_tensors
from parallel_ops.tensor_operations import replicate_tensor
from parallel_ops.tensor_operations import reduce_tensors
from parallel_ops.parallel_ops import parallel_ops
from algebraic_ops import algebraic_ops
from node_status import node_status
from node_types import node_types
import logging

from autograd_functions import Partition
from autograd_functions import Reduce
from autograd_functions import Replicate

import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor
from torch.distributed._tensor import Shard, Replicate, distribute_tensor

import os

logging.basicConfig(level=logging.DEBUG)
from torch import nn

class PCGNode:
    def __init__(self, id, input, type, machine_mapping, placements, parents) -> None:
        self.id = id
        self.input = input
        self.output = None
        self.type = type
        self.machine_mapping = machine_mapping
        self.placements = placements
        # self.local_tensor = local_tensor
        # self.dtensor = dtensor
        self.parents = parents
        self.status = node_status.WAITING if parents else node_status.READY
    
    def forward(self) -> None:

        if self.type == node_types.INPUT or self.type == node_types.WEIGHT or self.type == node_types.OUTPUT:
            # print(f"Changing {self.id} to {self.input[0]}")
            self.output = self.input[0]
            return
        
        if self.type == parallel_ops.PARTITION:
            distributed_tensor = distribute_tensor(self.input[0], self.machine_mapping, self.placements)
            # print(f"Changing {self.id} to {distributed_tensor}")
            self.output = distributed_tensor
        elif self.type == parallel_ops.REPLICATE:
            replicated_tensor = DTensor.from_local(self.input[0], self.machine_mapping, self.placements)
            # print(f"Changing {self.id} to {replicated_tensor}")
            self.output = replicated_tensor
        elif self.type == algebraic_ops.MATMUL:
            matmulres = torch.matmul(self.input[0], self.input[1])
            # print(f"Changing {self.id} to {matmulres}")
            self.output = matmulres
        # elif self.type == parallel_ops.COMBINE:



        # elif self.type == parallel_ops.REDUCE:



        
        










# class PCGNode:
#     def __init__(self, id, type, parents, machine_mapping=None, operation=None, dim=None, data=None) -> None:
#         self.id = id
#         self.type = type
#         self.parents = parents
#         self.machine_mapping = machine_mapping
#         self.operation = operation
#         self.dim = dim
#         self.data = data if data else []
#         self.outputs = None
#         self.grad = None
#         self.status = node_status.WAITING if parents else node_status.READY
        
        

    
#     def forward_pass(self) -> None:
#         # if self.type == node_types.INPUT or self.type == node_types.WEIGHT:
#         #     return
#         # if self.type == node_types.OUTPUT:
#         #     self.backward_status = node_status.READY
#         #     return
        
#         # if self.operation == parallel_ops.PARTITION:
#         #     self.data = partition_tensor(self.data[0][0], self.machine_mapping, self.dim)
#         # elif self.operation == parallel_ops.COMBINE:
#         #     self.data = combine_tensors(self.data[0], self.machine_mapping, self.dim)
#         # elif self.operation == parallel_ops.REPLICATE:
#         #     self.data = replicate_tensor(self.data[0][0], self.machine_mapping)
#         # elif self.operation == parallel_ops.REDUCE:
#         #     self.data = reduce_tensors(self.data[0], self.machine_mapping)
#         # elif self.operation == algebraic_ops.MATMUL:
#         #     self.data = [x @ y for x, y in zip(self.data[0], self.data[1])]
#         # elif self.operation == algebraic_ops.RELU:
#         #     self.data = [torch.relu(self.data[0][0])]

#         if self.type == node_types.INPUT or self.type == node_types.WEIGHT:
#             # self.outputs = [tensor.detach().requires_grad_() for tensor in self.data]
#             self.outputs = self.data

#         if self.type == node_types.OUTPUT:
#             self.outputs = self.data

#         if self.operation == parallel_ops.PARTITION:
#             self.outputs = Partition.apply(self.data[0][0], self.machine_mapping, self.dim)
#         elif self.operation == parallel_ops.REDUCE:
#             self.outputs = Reduce.apply(self.data[0], self.machine_mapping)

#         elif self.operation == parallel_ops.REPLICATE:
#             self.outputs = Replicate.apply(self.data[0][0], self.machine_mapping)
#         elif self.operation == algebraic_ops.MATMUL:
#             self.outputs = [x @ y for x, y in zip(self.data[0], self.data[1])]
#             for output in self.outputs:
#                 output.retain_grad()