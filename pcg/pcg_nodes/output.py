from pcg.pcg_nodes.pcg_node import PCGNode
import torch.distributed as dist
from pcg.pcg_nodes.parallel_tensor_attrs import *

class OutputNode(PCGNode):
    def __init__(
            self, 
            name: int, 
            parents: list, 
            machine_view: list,
            parallel_tensor_attrs: ParallelTensorAttrs=None
            ):
        super().__init__(
            name=name, 
            parents=parents, 
            parallel_tensor_attrs=parallel_tensor_attrs)
        self.machine_view = machine_view

    def forward(self, name_to_node: map):        
        self.data = name_to_node[self.parents[0]].data

        # print(f"{global_rank}-{self.name}: Output Done (Forward)")