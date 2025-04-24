from combine import CombineNode
from partition import PartitionNode
from replicate import ReplicateNode
from reduce import ReduceNode

import torch.distributed as dist
import torch
import os

import torch.nn as nn

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()

input_values_map = {}
if global_rank == 0:
    input_values_map["input"] = torch.tensor([[2, 3], [6, 7]], dtype=torch.float32).cuda(local_rank)

    weight = torch.tensor([[1, 2], [3, 4]], dtype=torch.float32).cuda(local_rank)
    input_values_map["weight"] = nn.Parameter(weight)

part_node = PartitionNode("partition",  ["input"], [0, 1], 0)
rep_node = ReplicateNode("replicate", ["weight"], [0, 1])
comb_node = CombineNode("combine", ["matmul"], [0, 1], 0)

if global_rank in part_node.machine_view:
    part_tensor = part_node.forward(input_values_map)
    input_values_map["partition"] = part_tensor

if global_rank in rep_node.machine_view:
    rep_tensor = rep_node.forward(input_values_map)
    input_values_map["replicate"] = rep_tensor

if global_rank in [0, 1]:
    matmul_tensor = torch.matmul(input_values_map["partition"], input_values_map["replicate"])
    input_values_map["matmul"] = matmul_tensor

if global_rank in comb_node.machine_view:
    comb_tensor = comb_node.forward(input_values_map)
    input_values_map["comb"] = comb_tensor

print(input_values_map["comb"])













    


