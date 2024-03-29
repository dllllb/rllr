import random
from collections import deque
from functools import partial
import torch

from ..utils import convert_to_torch


class ReplayBuffer:
    """
    Memory buffer for saving trajectories
    """

    def __init__(self, buffer_size: int, batch_size: int, device: torch.device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device

    def add(self, *args):
        self.buffer.append(args)

    def sample(self):
        batch = random.sample(self.buffer, k=self.batch_size)
        return map(partial(convert_to_torch, device=self.device), zip(*batch))

    def get(self):
        return map(partial(convert_to_torch, device=self.device), zip(*self.buffer))

    def clear(self):
        self.buffer.clear()

    def is_enough(self):
        return len(self.buffer) >= self.batch_size
