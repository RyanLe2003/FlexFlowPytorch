import torch

class PCGNode:
    def __init__(self, name, task, dependencies, machine_mapping) -> None:
        self.name = name
        self.task = task
        self.dependencies = dependencies
        self.machine_mapping = machine_mapping
        self.outputs = []
        self.status = "READY" if not self.dependencies else "WAITING"
    

def execute_node(node, results, index):
    torch_device = torch.device("cpu")
    if torch.cuda.is_available():
        device = node.machine_mapping[index]
        torch_device = torch.device(f"cuda:{device}")

    if node.task:
        inputs = []
        for parent in node.dependencies:
            for item in results[parent]:
                inputs.append(item.to(torch_device))
        node.outputs[index] = node.operation(*inputs)
    else:
        node.outputs[index] = None
    
    return node.outputs[index]