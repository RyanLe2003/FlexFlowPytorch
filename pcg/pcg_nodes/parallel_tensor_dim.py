from pcg.pcg_nodes.shard_parallel_dim import *
from pcg.pcg_nodes.replica_parallel_dim import *


class ParallelTensorDim:
    def __init__(self, shard_dims: list, replica_dims: ReplicaParallelDim) -> None:
        assert isinstance(shard_dims, list), "dims must be a list"
        assert all(isinstance(x, ShardParallelDim) for x in shard_dims), "all items must be MyClass instances"

        self.shard_dims = shard_dims
        self.replica_dims = replica_dims


        
        