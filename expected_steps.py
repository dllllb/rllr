import logging
import numpy as np
import torch

from copy import deepcopy
from torch import optim

from fast_tensor_data_loader import FastTensorDataLoader
from models import ExpectedStepsAmountRegressor

logger = logging.getLogger(__name__)


def _collect_episode(env, agent, explore=True):
    """
    A helper function for collect single episode
    """
    buffer = []
    state = env.reset()
    agent.explore = explore

    buffer.append(deepcopy(state))

    steps, done = 0, 0
    while not done:
        steps += 1
        action = agent.act(state, env.goal_state)
        state, _, done, _ = env.step(action)
        buffer.append(deepcopy(state))

    agent.reset_episode()
    env.close()

    buffer = [(x, state, steps - i) for i, x in enumerate(buffer)]
    return buffer


class ExpectedStepsAmountLearner:
    def __init__(self, grid_size, conf):
        self.model = ExpectedStepsAmountRegressor(grid_size, conf['model'])
        self.device = torch.device(conf['device'])
        self.model.to(self.device)
        self.model.device = self.device
        self.conf = conf
        self.buffer_size = conf['buffer_size']
        self.buffer = []
        self.epochs = conf['epochs']

    def collect_episodes(self, env, agent):
        for _ in range(self.buffer_size):
            self.buffer.extend(_collect_episode(env, agent))

    def learn(self, verbose=False):
        def _vstack(arr):
            arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
            return torch.from_numpy(arr).float()

        states, goal_states, y = map(_vstack, zip(*self.buffer))
        y = y.squeeze()

        loss_fn = torch.nn.L1Loss()
        train_loader = FastTensorDataLoader(states, goal_states, y, batch_size=self.conf['batch_size'], shuffle=True)
        optimizer = optim.Adam(self.model.parameters(), lr=self.conf['learning_rate'])
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

    def reset_buffer(self):
        self.buffer = []
