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


class RunningMeanStd(object):
    # https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    def __init__(self, epsilon=1e-4, shape=(), device='cpu'):
        self.mean = torch.zeros(shape, device=device)
        self.var = torch.ones(shape, device=device)
        self.count = epsilon

    def update(self, x):
        x = torch.tensor(x).float().view(-1, *self.mean.shape)

        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0)
        batch_count = x.shape[0]
        self.update_from_moments(batch_mean, batch_var, batch_count)

    def update_from_moments(self, batch_mean, batch_var, batch_count):
        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count
        m_a = self.var * (self.count)
        m_b = batch_var * (batch_count)
        M2 = m_a + m_b + torch.square(delta) * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        new_count = tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = new_count
