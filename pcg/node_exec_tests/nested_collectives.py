from pcg.pcg_nodes.partition import PartitionNode
from pcg.pcg_nodes.replicate import ReplicateNode
from pcg.pcg_nodes.combine import CombineNode
from pcg.pcg_nodes.reduce import ReduceNode
from pcg.pcg_nodes.input import InputNode

import torch.distributed as dist
import random
import torch
import os

local_rank = int(os.environ.get("LOCAL_RANK", 0))
dist.init_process_group(backend='nccl')
torch.cuda.set_device(local_rank)
global_rank = dist.get_rank()

input_data = []
for r in range(8):
    row = [r]
    # for c in range(8):
    #     row.append(random.random())
    input_data.append(row)

input_node = InputNode(1, [], [0], input_data)

exec_order = [input_node]
name_to_node = {1: input_node}
node_num = 2
while node_num < 5:
    part_node = PartitionNode(node_num, [node_num - 1], [0, 1], 0)
    exec_order.append(part_node)
    name_to_node[node_num] = part_node
    node_num += 1

while node_num < 8:
    comb_node = CombineNode(node_num, [node_num - 1], [0], 0)
    exec_order.append(comb_node)
    name_to_node[node_num] = comb_node
    node_num += 1

for node in exec_order:
    node.forward(name_to_node)
    print(f"{global_rank}: {node.data}")
