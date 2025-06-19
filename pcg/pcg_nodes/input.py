from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os
from pcg.pcg_nodes.parallel_tensor_attrs import *

# this will eventually need to be added when I get pcg
# would likely need to convert json to list to tensor
class InputNode(PCGNode):
    def __init__(
            self, 
            name: int, 
            parents: list, 
            parallel_tensor_attrs: ParallelTensorAttrs, 
            machine_view: list, 
            data):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        tensor = None
        if global_rank in machine_view:
            tensor = data
        
        super().__init__(
            name=name, 
            parents=parents, 
            parallel_tensor_attrs=parallel_tensor_attrs, 
            data=tensor)
        self.machine_view = machine_view

    def forward(self, name_to_node):
        pass


        



