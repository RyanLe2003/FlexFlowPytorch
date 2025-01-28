from enum import Enum

class parallel_ops(Enum):
    REDUCE = "reduce"
    PARTITION = "partition"
    REPLICATE = "replicate"
    COMBINE = "combine"

