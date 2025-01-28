from node_status import node_status
from parallel_ops.tensor_operations import partition_tensor
from parallel_ops.tensor_operations import combine_tensors
from parallel_ops.tensor_operations import replicate_tensor
from parallel_ops.tensor_operations import reduce_tensors
from parallel_ops.parallel_ops import parallel_ops
from algebraic_ops import algebraic_ops

class PCGNode:
    def __init__(self, type, data, machine_mapping, operation, dim, parents) -> None:
        self.type = type
        self.data = data
        self.machine_mapping = machine_mapping
        self.operation = operation
        self.dim = dim
        self.parents = parents
    
    def forward_pass(self) -> None:
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
        