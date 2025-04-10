from PCGNode import Partition, Replicate, Combine, Reduce
import torch.nn as nn
import torch
from concurrent.futures import ThreadPoolExecutor

class testModelWPartition(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand((4, 4)))
    
    @staticmethod
    def multiply(a, b):
        res = torch.matmul(a, b)
        return res

    
    def forward(self, x):
        parts = Partition.apply(x, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 1)
        parts2 = Partition.apply(self.weight, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 0)

        futures = []
        with ThreadPoolExecutor() as executor:
            for i in range(len(parts)):
                futures.append(executor.submit(self.multiply, parts[i], parts2[i]))
        
        # manual implementation of reduce
        # res = 0
        # for future in futures: 
        #     res += future.result().to("cuda:0")
        #     print(res)

        vals = []
        for future in futures:
            vals.append(future.result())

        res = Reduce.apply(vals, ["cuda:0"])
        
        return res

def PartitionTest():
    model = testModelWPartition()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()


    input_tensor = torch.rand((4, 4)).to(device)
    target = torch.rand((4, 4)).to(device)

    for epoch in range(20):
        optimizer.zero_grad()

        output = model(input_tensor)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

PartitionTest()

class testModel2(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand((4, 4)))
    
    @staticmethod
    def multiply(a, b):
        res = torch.matmul(a, b)
        return res
    
    def forward(self, x):
        parts = Partition.apply(x, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 0)
        dups = Replicate.apply(self.weight, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"])

        futures = []
        with ThreadPoolExecutor() as executor:
            for i in range(len(parts)):
                futures.append(executor.submit(self.multiply, parts[i], dups[i]))
        
        vals = []
        for future in futures:
            vals.append(future.result())

        res = Combine.apply(vals, ["cuda:0"], 0)
        return res
    
def Test2():
    model = testModel2()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()


    input_tensor = torch.rand((4, 4), requires_grad=True).to(device)
    target = torch.rand((4, 4)).to(device)

    for epoch in range(20):
        optimizer.zero_grad()

        output = model(input_tensor)

        loss = criterion(output, target)

        loss.backward()

        optimizer.step()
        
        print(f"Epoch {epoch}, Loss: {loss.item():.4f}")


# Test2()


