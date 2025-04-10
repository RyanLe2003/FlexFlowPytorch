from PCGNode import Partition
import torch.nn as nn
import torch

class testModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.rand((2, 3)))
    
    def forward(self, x):
        parts = Partition.apply(x, ["cuda:0", "cuda:1"], 0)
        parts2 = Partition.apply(self.weight, ["cuda:0", "cuda:1"], 1)

        mul1 = parts[0] * parts2[0]
        mul2 = parts[1] * parts2[1]

        mul1 = mul1.to("cuda:0")
        mul2 = mul2.to("cuda:0")

        res = mul1 + mul2
        return res

model = testModel()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.MSELoss()


input_tensor = torch.rand((3, 2)).to(device)
target = torch.rand((2, 2)).to(device)

for epoch in range(100):
    optimizer.zero_grad()

    output = model(input_tensor)

    loss = criterion(output, target)

    loss.backward()

    optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")