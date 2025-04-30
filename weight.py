from pcg_node import PCGNode
import torch
import torch.distributed as dist
import os

import torch.nn as nn

class WeightNode(PCGNode):
    def __init__(self, name, parents, machine_view, data):
        global_rank = dist.get_rank()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))

        tensor = None
        if global_rank in machine_view:
            tensor = nn.Parameter(torch.tensor(data, device=f'cuda:{local_rank}', dtype=torch.float32, requires_grad=True))

        super().__init__(name, parents, tensor)
        self.machine_view = machine_view

    def forward(self, name_to_node):
        pass
        


