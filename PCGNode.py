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

logging.basicConfig(level=logging.DEBUG)

import torch

class PCGNode:
    def __init__(self, id, type, parents, machine_mapping=None, operation=None, dim=None, data=None) -> None:
        self.id = id    
        self.type = type
        self.data = data if data else []
        self.machine_mapping = machine_mapping
        self.operation = operation
        self.dim = dim
        self.parents = parents
        self.status = node_status.WAITING if parents else node_status.READY

        self.grad = None
        self.backward_status = node_status.WAITING
    
    def forward_pass(self) -> None:
        if self.type == node_types.INPUT or self.type == node_types.WEIGHT:
            return
        if self.type == node_types.OUTPUT:
            self.backward_status = node_status.READY
            return
        
        if self.operation == parallel_ops.PARTITION:
            self.data = partition_tensor(self.data[0][0], self.machine_mapping, self.dim)
        elif self.operation == parallel_ops.COMBINE:
            self.data = combine_tensors(self.data[0], self.machine_mapping, self.dim)
        elif self.operation == parallel_ops.REPLICATE:
            self.data = replicate_tensor(self.data[0][0], self.machine_mapping)
        elif self.operation == parallel_ops.REDUCE:
            self.data = reduce_tensors(self.data[0], self.machine_mapping)
        elif self.operation == algebraic_ops.MATMUL:
            self.data = [x @ y for x, y in zip(self.data[0], self.data[1])]
    
    def backward_pass(self, pcg) -> None:
        if self.type == node_types.OUTPUT or self.grad == None:
            # For output nodes, initialize gradient as ones
            self.grad = [torch.ones_like(d) for d in self.data]
            
        if self.operation == parallel_ops.PARTITION:
            self.grad = combine_tensors(self.grad, self.machine_mapping, self.dim)
        elif self.operation == parallel_ops.COMBINE:
            self.grad = partition_tensor(self.grad, self.machine_mapping, self.dim)
        elif self.operation == parallel_ops.REPLICATE:
            self.grad = reduce_tensors(self.grad, self.machine_mapping)
        elif self.operation == parallel_ops.REDUCE:
            self.grad = replicate_tensor(self.grad[0], self.machine_mapping)
        elif self.operation == algebraic_ops.MATMUL:
            # Assuming self.parents[0] is left matrix and self.parents[1] is right matrix
            left_grad = [torch.matmul(g, p.t()) for g, p in zip(self.grad, pcg[self.parents[1]].data)]                                       
            right_grad = [torch.matmul(p.t(), g) for p, g in zip(pcg[self.parents[0]].data, self.grad)]
            if pcg[self.parents[0]].grad is None:
                pcg[self.parents[0]].grad = left_grad
            else:
                pcg[self.parents[0]].grad = [g1 + g2 for g1, g2 in zip(pcg[self.parents[0]].grad, left_grad)]
            
            if pcg[self.parents[1]].grad is None:
                pcg[self.parents[1]].grad = right_grad
            else:
                pcg[self.parents[1]].grad = [g1 + g2 for g1, g2 in zip(pcg[self.parents[1]].grad, right_grad)]
        
    def update_parameter(self):
        if self.grad is not None:
            # Simple SGD update
            learning_rate = 0.01
            self.data = [d - learning_rate * g for d, g in zip(self.data, self.grad)]
            self.grad = None  # Reset gradient after update