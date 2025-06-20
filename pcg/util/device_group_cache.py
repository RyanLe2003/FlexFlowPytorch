import torch.distributed as dist

cache = {}
def device_group_cache(machine_view):
    key = tuple(machine_view)
    if key not in cache:
        cache[key] = dist.new_group(machine_view)
    return cache[key]
