import random
from collections import deque
from functools import partial

from utils import convert_to_torch


class ReplayBuffer:
    """
    Memory buffer for saving trajectories
    """

    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device

    def add(self, *args):
        self.buffer.append(args)

    def sample(self):

        batch = random.sample(self.buffer, k=self.batch_size)
        return map(partial(convert_to_torch, device=self.device), zip(*batch))

    def is_enough(self):
        return len(self.buffer) > self.batch_size
