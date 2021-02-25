import logging
import numpy as np
import torch
import random

from collections import deque
from copy import deepcopy
from torch import optim

from fast_tensor_data_loader import FastTensorDataLoader
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

    def learn_(self, verbose=False):
        self.epochs = 10

        def _vstack(arr):
            if arr and isinstance(arr[0], dict):
                arr = [x['image'] for x in arr]
            arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
            return torch.from_numpy(arr).float()

        states, goal_states, y = map(_vstack, zip(*self.buffer))
        y = y.squeeze()

        loss_fn = torch.nn.L1Loss()
        train_loader = FastTensorDataLoader(states, goal_states, y, batch_size=self.config['batch_size'], shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'])
        self.model.train()

        loss_sum = 0
        for _ in range(self.epochs):
            for batch_state, batch_goal_state, batch_y in train_loader:
                output = self.model(batch_state.to(self.device), batch_goal_state.to(self.device))
                loss = loss_fn(output, batch_y.to(self.device))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loss_sum += loss.detach().cpu().numpy()

        if verbose:
            logger.info(f"Expected steps L1Loss:  {loss_sum / len(train_loader) / self.epochs}")

    def _collect_episode(self, env, agent, explore=True):
        """
        A helper function for collect single episode
        """
        self.max_steps = self.config['warm_up_max_steps']
        buffer = []
        state = env.reset()
        agent.explore = explore

        buffer.append(deepcopy(state))

        steps, done = 0, 0
        while not done and steps <= self.max_steps:
            steps += 1
            action = agent.act(state, env.goal_state)
            state, _, done, _ = env.step(action)
            buffer.append(deepcopy(state))

        env.close()

        buffer = [(x, state, steps - i) for i, x in enumerate(buffer)]
        return buffer

    def collect_episodes(self, env, agent):
        self.verbose = 1_000
        for i in range(self.config['warm_up']):
            if self.verbose and not i % self.verbose:
                logger.info(f"Collecting episodes:  {i}")
            self.buffer.extend(self._collect_episode(env, agent))
