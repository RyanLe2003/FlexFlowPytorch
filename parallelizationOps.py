from enum import Enum

class parallelizationOps(Enum):
    REDUCE = "reduce"
    PARTITION = "partition"
    REPLICATE = "replicate"
    COMBINE = "combine"

