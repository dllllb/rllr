import logging
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from ..buffer import ReplayBuffer
from ..utils import convert_to_torch
from .core import Algo

logger = logging.getLogger(__name__)


class DQN(Algo):
    """ An agent implementing Deep Q-Network algorithm"""

    def __init__(self,
                 qnetwork_local: torch.nn.Module,
                 qnetwork_target: torch.nn.Module,
                 replay_buffer: ReplayBuffer,
                 device: torch.device,
                 learning_rate: float = 0.001,
                 update_step: int = 4,
                 gamma: float = 0.9,
                 eps_start: float =  1.,
                 eps_end: float = 0.1,
                 eps_decay: float = 0.995,
                 tau: float = 0.001,
                 explore: bool = True):
        """ Initializes a DQN agent.

        Params:
        -------
            qnetwork_local: qnetwork for local updates q-values
            qnetwork_target: a copy of qnetwork_local for smooth q function
            replay_buffer: buffer containing experience <s,a, r, s', d> tuples
            device: cpu or gpu device
            learning_rate: learning rate
            update_step: number of steps between policy updates
            gamma: reward discount factor
        """
        self.device = device
        logger.info("Running on device: {}".format(self.device))
        self.qnetwork_local = qnetwork_local.to(self.device)
        self.qnetwork_target = qnetwork_target.to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=learning_rate)
        self.action_size = self.qnetwork_local.output_size
        self.replay_buffer = replay_buffer
        self.batch_size = replay_buffer.batch_size
        self.buffer_size = replay_buffer.buffer_size
        self.step = 0
        self.eps = eps_start
        self.eps_decay = eps_decay
        self.eps_end = eps_end
        self.explore = explore
        self.tau = tau
        self.gamma = gamma
        self.update_step = update_step

    def reset_episode(self):
        """
        Resets episode and update epsilon decay
        """
        self.eps = max(self.eps_end, self.eps_decay * self.eps)

    def _learn(self):
        """
        Learns values-actions network

        """
        # Sample batch from replay buffer
        states, next_states, actions, rewards, dones = self.replay_buffer.sample()

        with torch.no_grad():
            values = self.qnetwork_target.forward(next_states)
        targets = rewards + self.gamma * values.max(1)[0].view(dones.size()) * (1 - dones)
        outputs = self.qnetwork_local.forward(states).gather(1, actions.long())
        self.optimizer.zero_grad()
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def update(self, state, action, reward, next_state, done):
        """
        Makes an update step of algorithm and append sars to buffer replay

        Params:
        -------
        state - current state
        action - action made
        reward - reward for an action
        next_state - next state from env
        done - episode finishing flag
        """
        self.replay_buffer.add(state, next_state, action, reward, float(done))

        self.step = (self.step + 1) % self.update_step
        if self.step == 0:
            if self.replay_buffer.is_enough():
                self._learn()
                self._reset_target_network()

    def _reset_target_network(self):
        """
        Resets params of target network to values from local network
        """
        params = zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters())
        for target_param, local_param in params:
            updated_params = self.tau * local_param.data + (1 - self.tau) * target_param.data
            target_param.data.copy_(updated_params)

    def act(self, state):
        """
        Selects action from state if epsilon-greedy way

        Params:
        state - current state

        """
        self.step += 1

        if self.explore and random.random() < self.eps:
            return random.choice(np.arange(self.action_size))
        else:
            self.qnetwork_local.eval()
            with torch.no_grad():
                states = convert_to_torch([state], self.device)
                action_values = self.qnetwork_local(states)

            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
