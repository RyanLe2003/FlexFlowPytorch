import torch
from pcg.pcg_nodes.parallel_tensor_attrs import *

class PCGNode:
    def __init__(
            self, 
            name: int, 
            parents: list,
            parallel_tensor_attrs: ParallelTensorAttrs=None, 
            data=None,
            ):
        self.name = name
        self.parents = parents
        self.parallel_tensor_attrs = parallel_tensor_attrs
        self.data = data

    def forward(self):
        raise NotImplementedError
    
    def get_shape(self):
        shape = []
        shard_dims = (self.parallel_tensor_attrs
                      .tensor_shape
                      .tensor_dim
                      .shard_dims)
        
        for shard_dim in shard_dims:
            shape.append(shard_dim.size)
        
        return shape

    