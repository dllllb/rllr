import torch
from collections import deque


class EpisodicMemory:
    """The Episodic Memory Module of
        [Badia et al. (2020)](https://openreview.net/forum?id=Sye57xStvB)
    as described in the paper.
    """
    def __init__(self,
                 maxlen=30_000,
                 k=8,
                 kernel_cluster_distance=0.008,
                 kernel_epsilon=0.0001,
                 c=0.001,
                 sm=8,
                 gamma=0.99):
        self.memory = deque([], maxlen=maxlen)
        self.k = k
        self.kernel_cluster_distance = kernel_cluster_distance
        self.kernel_epsilon = kernel_epsilon
        self.c = c
        self.sm = sm
        self.gamma = gamma
        self.dm = 0

    def add(self, state: torch.Tensor):
        self.memory.append(state)

    def compute_reward(self, states: torch.Tensor) -> torch.Tensor:
        if len(self.memory) < self.k:
            return torch.zeros(len(states))

        with torch.no_grad():
            dist = torch.cdist(torch.stack(tuple(self.memory), dim=0), states)
            dk, _ = dist.topk(self.k, largest=False, dim=0)

            self.dm = self.gamma * self.dm + (1 - self.gamma) + dk.mean()
            dn = dk / self.dm

            dn = torch.clamp(dn - self.kernel_cluster_distance, min=0)
            kernel = self.kernel_epsilon / (dn + self.kernel_epsilon)
            s = torch.sqrt(kernel.sum(dim=0)) + self.c
            return torch.where(s > self.sm, torch.zeros_like(s), 1 / s)

    def clear(self):
        self.memory.clear()
