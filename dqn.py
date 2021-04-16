import logging
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from replay import ReplayBuffer

from utils import convert_to_torch

logger = logging.getLogger(__name__)


class DQNAgentGoal:
    """ An agent implementing Deep Q-Network algorithm"""

    def __init__(self, qnetwork_local, qnetwork_target, action_size, config):
        """Initializes an Agent.

        Params:
        -------
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.device = torch.device(config['device'])
        logger.info("Running on device: {}".format(self.device))
        self.qnetwork_local = qnetwork_local.to(self.device)
        self.qnetwork_target = qnetwork_target.to(self.device)

        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config['learning_rate'])
        self.action_size = action_size
        self.replay_buffer = ReplayBuffer(config['buffer_size'], config['batch_size'], self.device)
        self.step = 0
        self.eps = config['eps_start']
        self.explore = True
        self.config = config

    def reset_episode(self):
        """
        Resets episode and update epsilon decay
        """
        self.eps = max(self.config['eps_end'], self.config['eps_decay'] * self.eps)

    def learn(self):
        """
        Learns values-actions network

        """
        # Sample batch from replay buffer
        states, next_states, actions, rewards, dones = self.replay_buffer.sample()

        with torch.no_grad():
            values = self.qnetwork_target.forward(next_states)
        targets = rewards + self.config['gamma'] * values.max(1)[0].view(dones.size()) * (1 - dones)
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

        self.step = (self.step + 1) % self.config['update_step']
        if self.step == 0:
            if self.replay_buffer.is_enough():
                self.learn()
                self.reset_target_network()

    def reset_target_network(self):
        """
        Resets params of target network to values from local network
        """
        params = zip(self.qnetwork_target.parameters(), self.qnetwork_local.parameters())
        for target_param, local_param in params:
            updated_params = self.config['tau'] * local_param.data + (1 - self.config['tau']) * target_param.data
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


def get_dqn_agent(config, get_policy_function, action_size):
    qnetwork_local = get_policy_function()
    qnetwork_target = get_policy_function()
    qnetwork_target.load_state_dict(qnetwork_local.state_dict())

    agent = DQNAgentGoal(
        qnetwork_local=qnetwork_local,
        qnetwork_target=qnetwork_target,
        action_size=action_size,
        config=config
    )
    return agent
