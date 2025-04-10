from PCGNode import Partition
import torch.nn as nn
import torch
from concurrent.futures import ThreadPoolExecutor

class testModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand((4, 4)))
    
    @staticmethod
    def multiply(a, b):
        res = a * b
        print(res)
        res = res.to("cuda:0")
        return res

    
    def forward(self, x):
        parts = Partition.apply(x, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 0)
        parts2 = Partition.apply(self.weight, ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], 1)

        futures = []
        with ThreadPoolExecutor() as executor:
            for i in range(len(parts)):
                futures.append(executor.submit(self.multiply, parts[i], parts2[i]))
        
        res = 0
        for future in futures:
            res += future.result()
        
        return res

model = testModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


input_tensor = torch.rand((4, 4)).to(device)
target = torch.rand((4, 4)).to(device)

for epoch in range(2):
    optimizer.zero_grad()

    output = model(input_tensor)

    loss = criterion(output, target)

    loss.backward()

    optimizer.step()

    # print(f"Epoch {epoch}, Loss: {loss.item():.4f}")