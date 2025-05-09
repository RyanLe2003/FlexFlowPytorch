from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os
import torch.nn as nn

class SoftMaxNode(PCGNode):
    def __init__(self, name, parents, machine_view, dim):
        super().__init__(name, parents)
        self.machine_view = machine_view
        self.dim = dim

    def forward(self, name_to_node):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        val = name_to_node[self.parents[0]].data[0]

        if global_rank not in self.machine_view:
            self.data = [val]
            # print(f"{global_rank}-{self.name}: Softmax Done")
            return

        m = nn.Softmax(self.dim)
        self.data = [m(val)]

        # print(f"{global_rank}-{self.name}: Softmax Done")



    