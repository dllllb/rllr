import logging
import numpy as np
import torch
import random

from collections import deque
from torch import optim

from models import ExpectedStepsAmountRegressor

logger = logging.getLogger(__name__)


class ExpectedStepsAmountLeaner:
    def __init__(self, config):
        self.model = ExpectedStepsAmountRegressor(config['model'])
        self.device = torch.device(config['device'])
        self.model.to(self.device)
        self.model.device = self.device
        self.config = config
        self.buffer_size = config['buffer_size']
        self.epochs = config['epochs']
        self.buffer = deque(maxlen=self.buffer_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.step = 0

    def _convert_to_torch(self, arr):
        if arr and isinstance(arr[0], dict):
            arr = [x['image'] for x in arr]
        arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
        return torch.from_numpy(arr).float().to(self.device)

    def update(self, state, goal_state, steps):
        self.buffer.append((state, goal_state, steps))

        self.step = (self.step + 1) % self.config['update_step']
        if self.step == 0:
            if len(self.buffer) > self.config['batch_size']:
                return self.learn()

    def learn(self, verbose=False):
        if self.config['batch_size'] > len(self.buffer): return
        self.model.train()
        loss_sum = 0.
        loss_fn = torch.nn.L1Loss()
        for i in range(self.epochs):
            # Sample batch from replay buffer
            states, goal_states, y = self.sample_batch()

            output = self.model(states, goal_states)
            loss = loss_fn(output, y)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_sum += loss.cpu().item()

            if verbose:
                logger.info(f"Expected steps L1Loss: {loss_sum / self.epochs}")

        return loss_sum / self.epochs

    def sample_batch(self):
        """
        Samples a batch of experience from replay buffer random uniformily
        """
        batch = random.sample(self.buffer, k=self.config['batch_size'])
        states, goal_states, y = map(self._convert_to_torch, zip(*batch))
        y = y.squeeze()

        return states, goal_states, y

    def reset_buffer(self):
        self.buffer = deque(maxlen=self.buffer_size)
