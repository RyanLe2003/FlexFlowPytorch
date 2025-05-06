from pcg.pcg_nodes.pcg_node import PCGNode
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

        proc_one = name_to_node[self.parents[0]].data
        proc_two = name_to_node[self.parents[1]].data

        if (len(proc_one) != len(proc_two)):
            raise RuntimeError("Non matching data length in matmul")
        
        new_data = []
        for i in range(len(proc_one)):
            res = torch.matmul(proc_one[i], proc_two[i])
            new_data.append(res)
        
        self.data = new_data