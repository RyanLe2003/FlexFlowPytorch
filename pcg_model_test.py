import torch
from parse_graph import parse_graph
from pcg_model import PCGModel
import torch.nn as nn

pcg1 = [
    {
        "name" : "input",
        "type" : "Input",
        "value" : torch.rand((4, 4))
    },
    {
        "name" : "partition1",
        "type" : "Partition",
        "parents" : ["input"],
        "machine_mapping" : ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "dim": 1
    },
    {
        "name" : "weight1",
        "type" : "Weight",
        "value" : torch.rand((4, 4))
    },
    {
        "name" : "partition2",
        "type" : "Partition",
        "parents" : ["weight1"],
        "machine_mapping" : ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        "dim": 0
    },
    {
        "name" : "matmul1",
        "type" : "Matmul",
        "parents" : ["partition1", "partition2"],
        "machine_mapping" : ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
    },
    {
        "name" : "reduce1",
        "type" : "Reduce",
        "parents" : ["matmul1"],
        "machine_mapping" : ["cuda:0"]
    },
    {
        "name" : "OUTPUT",
        "type" : "Output",
        "parents" : ["reduce1"]
    }
]

input, weights, node_map, dependency_graph = parse_graph(pcg1)

model = PCGModel(node_map, weights, dependency_graph)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
target = torch.rand((4, 4)).to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()

for epoch in range(10):
    optimizer.zero_grad()
    output = model(input)

    loss = criterion(output, target)
    loss.backward()

    optimizer.step()
    
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")




