from old_comb import CombineNode
from old_part import PartitionNode
from old_rep import ReplicateNode
from reduce import ReduceNode

import torch.distributed as dist
import torch
import os

import torch.nn as nn

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()

print(f"Process initialized: global_rank={global_rank}, local_rank={local_rank}")

input_values_map = {}
if global_rank == 0:
    print(f"Rank {global_rank}: Creating input tensors")
    input = torch.tensor([[2, 3], [6, 7]], dtype=torch.float32, requires_grad=True).cuda(local_rank)
    input_values_map["input"] = input

    weight = nn.Parameter(torch.tensor([[1, 2], [3, 4]], dtype=torch.float32, requires_grad=True).cuda(local_rank))
    input_values_map["weight"] = weight

    print(f"Weight: {weight}")

    optimizer = torch.optim.SGD([weight], lr=0.01)

print(f"Rank {global_rank}: Setting up nodes")
part_node = PartitionNode("partition",  ["input"], [0, 1], 0)
rep_node = ReplicateNode("replicate", ["weight"], [0, 1])
comb_node = CombineNode("combine", ["matmul"], [0, 1], 0)


# for part, combine, matmul this needs to be if in node machine view and parent machine_view because parent view provides source
if global_rank in part_node.machine_view:
    part_tensor = part_node.forward(input_values_map)
    print(f"{global_rank}: {part_tensor}")
    input_values_map["partition"] = part_tensor

if global_rank in rep_node.machine_view:
    rep_tensor = rep_node.forward(input_values_map)
    print(rep_tensor)
    input_values_map["replicate"] = rep_tensor

if global_rank in [0, 1]:
    matmul_tensor = torch.matmul(input_values_map["partition"], input_values_map["replicate"])
    print(f"{global_rank}: {matmul_tensor}")
    input_values_map["matmul"] = matmul_tensor

if global_rank in comb_node.machine_view:
    comb_tensor = comb_node.forward(input_values_map)

    print(f"{global_rank}: {comb_tensor}")
    input_values_map["comb"] = comb_tensor


if global_rank in [0, 1]:
    target = torch.tensor([[10.0, 20.0], [30.0, 40.0]], dtype=torch.float32).cuda(local_rank)
    prediction = input_values_map["comb"]

    loss_fn = nn.MSELoss()
    loss = loss_fn(prediction, target)

    loss.backward()

print(f"{global_rank}: Name: {part_tensor.name} Leaf: {part_tensor.is_leaf}, requires_grad: {part_tensor.requires_grad}, grad: {part_tensor.data.grad}")
print(f"{global_rank}: Name: {rep_tensor.name} Leaf: {rep_tensor.is_leaf}, requires_grad: {rep_tensor.requires_grad}, grad: {rep_tensor.data.grad}")
print(f"{global_rank}: Name: {matmul_tensor.name} Leaf: {matmul_tensor.is_leaf}, requires_grad: {matmul_tensor.requires_grad}, grad: {matmul_tensor.data.grad}")
print(f"{global_rank}: Name: {comb_tensor.name} Leaf: {comb_tensor.is_leaf}, requires_grad: {comb_tensor.requires_grad}, grad: {comb_tensor.data.grad}")


if global_rank in [0]:
    print(f"{global_rank}: Name: Input Leaf: {input.is_leaf}, requires_grad: {input.requires_grad}, grad: {input.grad}")
    print(f"{global_rank}: Name: Weight Leaf: {weight.is_leaf}, requires_grad: {weight.requires_grad}, grad: {weight.grad}")

    optimizer.step()
    optimizer.zero_grad()

    print(input_values_map["weight"])

print(f"Rank {global_rank}: Script completed")




                             














    


