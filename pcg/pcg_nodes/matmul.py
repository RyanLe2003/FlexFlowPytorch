from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os
from pcg.pcg_nodes.parallel_tensor_attrs import *
from pcg.util.check_dist import get_rank

class MatmulNode(PCGNode):
    def __init__(
            self, 
            name: int, 
            parents: list,
            machine_view: list,
            parallel_tensor_attrs: ParallelTensorAttrs=None):
        super().__init__(
            name=name, 
            parents=parents,
            parallel_tensor_attrs=parallel_tensor_attrs)
        self.machine_view = machine_view

    def forward(self, name_to_node: map):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = get_rank()

        if global_rank not in self.machine_view:
            return

        proc_one = name_to_node[self.parents[0]].data
        proc_two = name_to_node[self.parents[1]].data

        self.data = torch.matmul(proc_one, proc_two)