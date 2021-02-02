import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from collections import deque
import random

BATCH_SIZE = 128  # 128
UPDATE_STEP = 4  # 4
BUFFER_SIZE = 100_000  # 100_000
LEARNING_RATE = 1e-3  # 5e-3
GAMMA = 0.9  # 0.9
EPS_START = 1  # 1
EPS_END = 0.1  # 0.1
EPS_DECAY = 0.995  # 0.995
TAU = 1e-3  # 1e-3


class DQNAgentGoal:
    """ An agent implementing Deep Q-Network algorithm"""

    def __init__(self, qnetwork_local, qnetwork_target, action_size, cuda=True):
        """Initializes an Agent.

        Params:
        -------
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.device = torch.device("cuda:0" if cuda else "cpu")
        print("Running on device: {}".format(self.device))
        self.qnetwork_local = qnetwork_local.to(self.device)
        self.qnetwork_target = qnetwork_target.to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LEARNING_RATE)
        self.action_size = action_size
        self.buffer = deque(maxlen=BUFFER_SIZE)
        self.step = 0
        self.eps = EPS_START
        self.explore = True

    def reset_episode(self):
        """
        Resets episode and update epsilon decay
        """
        self.eps = max(EPS_END, EPS_DECAY * self.eps)

    def learn(self):
        """
        Learns values-actions network

        """
        # Sample batch from replay buffer
        states, next_states, goal_states, actions, rewards, dones = self.sample_batch()

        values = self.qnetwork_target.forward(next_states, goal_states).detach()
        targets = rewards + GAMMA * values.max(1)[0].view(dones.size()) * (1 - dones)
        outputs = self.qnetwork_local.forward(states, goal_states).gather(1, actions.long())
        self.optimizer.zero_grad()
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def update(self, state, goal_state, action, reward, next_state, done):
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
        self.buffer.append((state, next_state, goal_state, action, reward, float(done)))

        self.step = (self.step + 1) % UPDATE_STEP
        if self.step == 0:
            if len(self.buffer) > BATCH_SIZE:
                self.learn()
                self.reset_target_network()

    def reset_target_network(self):
        """
        Resets params of target network to values from local network
        """
        params = zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters())
        for target_param, local_param in params:
            updated_params = TAU * local_param.data + (1 - TAU) * target_param.data
            target_param.data.copy_(updated_params)

    def _vstack(self, arr):
        arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
        return torch.from_numpy(arr).float().to(self.device)

    def act(self, state, goal_state):
        """
        Selects action from state if epsilon-greedy way

        Params:
        state - current state

        """
        self.step += 1

        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(self._vstack([state]), self._vstack([goal_state]))
        self.qnetwork_local.train()

        if (random.random() > self.eps) and self.explore:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def sample_batch(self):
        """
        Samples a batch of experience from replay buffer random uniformily
        """
        batch = random.sample(self.buffer, k=BATCH_SIZE)

        states, next_states, goal_states, actions, rewards, dones = map(self._vstack, zip(*batch))

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return states, next_states, goal_states, actions, rewards, dones
