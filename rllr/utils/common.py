import torch
import numpy as np


def switch_reproducibility_on(seed=42):
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_to_torch(arr, device='cpu'):
    if arr and isinstance(arr[0], dict):
        res = {
            key: convert_to_torch([x[key] for x in arr], device=device) for key in arr[0].keys()
        }
        return res

    else:
        res = np.vstack([np.expand_dims(x, axis=0) for x in arr])
        return torch.from_numpy(res).float().to(device)
