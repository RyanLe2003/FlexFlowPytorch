import torch
print(torch.__version__)
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed._tensor import DTensor
from torch.distributed._tensor import Shard, Replicate, distribute_tensor

import os

local_rank = int(os.environ.get("LOCAL_RANK", 0))
torch.cuda.set_device(local_rank)

torch.distributed.init_process_group(backend="nccl")
mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("dp", "tp"))

local_weight = torch.randn(4, 2, device="cuda", requires_grad=True)
local_input = torch.randn(1, 4, device="cuda")

# # partition test
# device_mesh = init_device_mesh("cuda", (2, ))
# tensor = torch.tensor([[2, 3], [6, 7]], dtype=torch.float32)
# distributed_tensor = distribute_tensor(tensor, device_mesh=device_mesh, placements=[Shard(0)])

weight_dt = DTensor.from_local(local_weight, mesh, [Shard(0), Replicate()])
weight_dt.requires_grad
input_dt = DTensor.from_local(local_input, mesh, [Replicate(), Shard(1)])

output_dt = torch.matmul(input_dt, weight_dt)
local_target = torch.randn_like(output_dt.to_local())
target_dt = DTensor.from_local(local_target, mesh, output_dt.placements)

learning_rate = 0.01

# print(f"Before Weight: {weight_dt}")

for epoch in range(1):
    if local_weight.grad is not None:
        local_weight.grad.zero_()

    # Forward pass
    output_dt = torch.matmul(input_dt, weight_dt)
    
    # compute loss
    loss_dt = ((output_dt - target_dt) ** 2).mean()
    loss = loss_dt.to_local()
    
    # Backward pass
    loss.backward()

    # print(f"weight: {weight_dt}")

    # synchronize for replicates (reduce - does this under the hood)
    # dist.all_reduce(local_weight.grad.data, op=dist.ReduceOp.SUM)
    # local_weight.grad.data /= dist.get_world_size()  

    # print(f"weight after sync: {weight_dt}")
    
    # Manual gradient update
    with torch.no_grad():
        local_weight.data -= learning_rate * local_weight.grad

# combine test
# final_weight_dt = weight_dt.redistribute(placements=[Replicate()])
# print(f"Final Weight (Replicated): {final_weight_dt.to_local()}")