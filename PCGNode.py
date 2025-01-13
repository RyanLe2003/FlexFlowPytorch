import torch

class PCGNode:
    def __init__(self, name, task, dependencies, machine_mapping) -> None:
        self.name = name
        self.task = task
        self.status = "READY" if not dependencies else "WAITING"
        self.dependencies = dependencies  # name of parent nodes
        self.machine_mapping = machine_mapping
        self.data = []


def execute_node(node, parent_outputs, index):
    torch_device = torch.device("cpu")
    if torch.cuda.is_available():
        device = node.machine_mapping[index]
        torch_device = torch.device(f"cuda:{device}")

    if node.task:
        inputs = []
        inputs.append(parent_outputs[index].to(torch_device))
        return node.operation(*inputs)
    else:
        return None