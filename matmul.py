from pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

class MatmulNode(PCGNode):
    def __init__(self, name, parents, machine_view):
        super().__init__(name, parents)
        self.machine_view = machine_view

    def forward(self, name_to_node):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        if global_rank not in self.machine_view:
            return
        
        tensor_one = name_to_node[self.parents[0]].data
        tensor_two = name_to_node[self.parents[1]].data

        self.data = torch.matmul(tensor_one, tensor_two)