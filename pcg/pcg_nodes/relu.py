from pcg.pcg_nodes.pcg_node import PCGNode
import torch
import torch.distributed as dist
import os
import torch.nn as nn

class ReluNode(PCGNode):
    def __init__(self, name, parents, machine_view):
        super().__init__(name, parents)
        self.machine_view = machine_view

    def forward(self, name_to_node):
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        global_rank = dist.get_rank()

        if global_rank not in self.machine_view:
            return
        
        val = name_to_node[self.parents[0]].data[0]

        if global_rank not in self.machine_view:
            self.data = [val]
            # print(f"{global_rank}-{self.name}: RELU Done")
            return

        m = nn.ReLU()
        self.data = [m(val)]

        # print(f"{global_rank}-{self.name}: RELU Done")



    