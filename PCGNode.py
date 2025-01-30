from node_status import node_status
from parallel_ops.tensor_operations import partition_tensor
from parallel_ops.tensor_operations import combine_tensors
from parallel_ops.tensor_operations import replicate_tensor
from parallel_ops.tensor_operations import reduce_tensors
from parallel_ops.parallel_ops import parallel_ops
from algebraic_ops import algebraic_ops
from node_status import node_status
from node_types import node_types

class PCGNode:
    def __init__(self, id, type, parents, machine_mapping=None, operation=None, dim=None, data=None) -> None:
        self.id = id    
        self.type = type
        self.data = data
        self.machine_mapping = machine_mapping
        self.operation = operation
        self.dim = dim
        self.parents = parents
        self.status = node_status.WAITING if parents else node_status.READY
    
    def forward_pass(self) -> None:
        if self.type == node_types.INPUT:
            return
        if self.type == node_types.OUTPUT:
            if len(self.parents) == 0:
                return
            self.data = self.parents[0].data
            return
        
        if self.operation == parallel_ops.PARTITION:
            self.data = partition_tensor(self.data, self.machine_mapping, self.dim)
        elif self.operation == parallel_ops.COMBINE:
            self.data = combine_tensors(self.data, self.machine_mapping, self.dim)
        elif self.operation == parallel_ops.REPLICATE:
            self.data = replicate_tensor(self.data, self.machine_mapping)
        elif self.operation == parallel_ops.REDUCE:
            self.data = reduce_tensors(self.data, self.machine_mapping)
        elif self.operation == algebraic_ops.MATMUL:
            self.data = [x @ y for x, y in zip(self.parents[0].data, self.parents[1].data)]
        