import torch
from parse_graph import parse_graph
from pcg_model import PCGModel
import torch.nn as nn

def test1():
    torch.manual_seed(42)
    pcg = [
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

    input, weights, node_map, dependency_graph = parse_graph(pcg)

    model = PCGModel(node_map, weights, dependency_graph)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = torch.rand((4, 4)).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    output = None
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        # print(f"Output at {epoch}: {output}")
    
    print(f"Final output: {output}")

def test_non_parallel_12():
    torch.manual_seed(42)

    pcg = [
        {
            "name" : "input",
            "type" : "Input",
            "value" : torch.rand((4, 4))
        },
        {
            "name" : "weight1",
            "type" : "Weight",
            "value" : torch.rand((4, 4))
        },
        {
            "name" : "matmul1",
            "type" : "Matmul",
            "parents" : ["input", "weight1"],
            "machine_mapping" : ["cuda:0"],
        },
        {
            "name" : "OUTPUT",
            "type" : "Output",
            "parents" : ["matmul1"]
        }
    ]

    input, weights, node_map, dependency_graph = parse_graph(pcg)

    model = PCGModel(node_map, weights, dependency_graph)
    device = torch.device("cpu") # this is not robust!!! make sure to implement a way for algebraic operations to move to/from devices
    target = torch.rand((4, 4)).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    output = None

    for epoch in range(10):
        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        # print(f"Output at {epoch}: {output}")
    
    print(f"Final output: {output}")


def test2():
    torch.manual_seed(42)
    pcg2 = [
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
            "dim": 0
        },
        {
            "name" : "weight1",
            "type" : "Weight",
            "value" : torch.rand((4, 4))
        },
        {
            "name" : "replicate1",
            "type" : "Replicate",
            "parents" : ["weight1"],
            "machine_mapping" : ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        },
        {
            "name" : "matmul1",
            "type" : "Matmul",
            "parents" : ["partition1", "replicate1"],
            "machine_mapping" : ["cuda:0", "cuda:1", "cuda:2", "cuda:3"],
        },
        {
            "name" : "combine1",
            "type" : "Combine",
            "parents" : ["matmul1"],
            "machine_mapping" : ["cuda:0"],
            "dim" : 0
        },
        {
            "name" : "OUTPUT",
            "type" : "Output",
            "parents" : ["combine1"]
        }
    ]   
    input, weights, node_map, dependency_graph = parse_graph(pcg2)

    model = PCGModel(node_map, weights, dependency_graph)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = torch.rand((4, 4)).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    output = None
    for epoch in range(10):
        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
        # print(f"Output at {epoch}: {output}")
    
    print(f"Final output: {output}")


two_layer_MLP = [
    {
        "name" : "input",
        "type" : "Input",
        "value" : torch.rand((4, 4))
    },
    {
        "name": "partition1",
        "type": "Partition",
        "parents" : ["input"],
        "machine_mapping" : ["cuda:0", "cuda:1"],
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
        "machine_mapping" : ["cuda:0", "cuda:1"],
        "dim": 0
    },
    {
        "name" : "matmul1",
        "type" : "Matmul",
        "parents" : ["partition1", "partition2"],
        "machine_mapping" : ["cuda:0", "cuda:1"],
    },
    {
        "name" : "reduce1",
        "type" : "Reduce",
        "parents" : ["matmul1"],
        "machine_mapping" : ["cuda:0"]
    },
    {
        "name": "replicate1",
        "type" : "Replicate",
        "parents": ["reduce1"],
        "machine_mapping": ["cuda:0", "cuda:1"]
    },
    {
        "name" : "relu1",
        "type" : "ReLU",
        "parents" : ["replicate1"],
        "machine_mapping": ["cuda:0", "cuda:1"]
    },
    {
        "name" : "weight2",
        "type" : "Weight",
        "value" : torch.rand((4, 2)) 
    },
    {
        "name" : "partition3",
        "type" : "Partition",
        "parents" : ["weight2"],
        "machine_mapping" : ["cuda:0", "cuda:1"],
        "dim": 1
    },
    {
        "name" : "matmul2",
        "type" : "Matmul",
        "parents" : ["relu1", "partition3"],
        "machine_mapping" : ["cuda:0", "cuda:1"],
    },
    {
        "name" : "relu2",
        "type" : "ReLU",
        "parents" : ["matmul2"],
        "machine_mapping": ["cuda:0", "cuda:1"]
    },
    {
        "name" : "combine1",
        "type" : "Combine",
        "parents" : ["relu2"],
        "machine_mapping": ["cuda:0"],
        "dim" : 1
    },
    {
        "name" : "OUTPUT",
        "type" : "Output",
        "parents" : ["combine1"]
    }
]

def test3():
    input, weights, node_map, dependency_graph = parse_graph(two_layer_MLP)

    print(dependency_graph)

    model = PCGModel(node_map, weights, dependency_graph)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    target = torch.rand((4, 2)).to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for epoch in range(50):
        optimizer.zero_grad()
        output = model(input)

        loss = criterion(output, target)
        loss.backward()

        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# print("test1")
# test1()
# print("-----------------------------")
# print("test2")
# test2()
# print("-----------------------------")
# print("non parallel for test 1 and 2")
# test_non_parallel_12()
print("-----------------------------")
print("test3")
test3()



