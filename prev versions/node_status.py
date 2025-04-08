from enum import Enum

class node_status(Enum):
    COMPLETED = "COMPLETED"
    READY = "READY"
    WAITING = "WAITING"
    RUNNING = "RUNNING"