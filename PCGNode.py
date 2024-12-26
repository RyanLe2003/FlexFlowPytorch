class PCGNode:
    def __init__(self, type, task) -> None:
        self.type = type
        self.task = task
        self.inputs = []
        self.outputs = []