import torch
from pcg.pcg_nodes.parallel_tensor_attrs import *

class PCGNode:
    def __init__(self, name, parents, parallel_tensor_attrs: ParallelTensorAttrs, data=None):
        self.name = name
        self.parents = parents
        self.data = []
        self.data.append(data)

        self.parallel_tensor_attrs = parallel_tensor_attrs

    def forward(self):
        raise NotImplementedError

    