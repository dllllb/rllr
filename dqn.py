import logging
import numpy as np
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from collections import deque

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
        self.buffer = deque(maxlen=config['buffer_size'])
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
        states, next_states, goal_states, actions, rewards, dones = self.sample_batch()

        values = self.qnetwork_target.forward(next_states, goal_states).detach()
        targets = rewards + self.config['gamma'] * values.max(1)[0].view(dones.size()) * (1 - dones)
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

        self.step = (self.step + 1) % self.config['update_step']
        if self.step == 0:
            if len(self.buffer) > self.config['batch_size']:
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

        if self.explore and random.random() < self.eps:
            return random.choice(np.arange(self.action_size))
        else:
            return np.argmax(action_values.cpu().data.numpy())

    def sample_batch(self):
        """
        Samples a batch of experience from replay buffer random uniformily
        """
        batch = random.sample(self.buffer, k=self.config['batch_size'])

        states, next_states, goal_states, actions, rewards, dones = map(self._vstack, zip(*batch))

        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return states, next_states, goal_states, actions, rewards, dones

    def save_model(self, file):
        torch.save(self.qnetwork_target, file)


def get_dqn_agent(config, get_net_function, action_size):
    qnetwork_local = get_net_function()
    qnetwork_target = get_net_function()
    qnetwork_target.load_state_dict(qnetwork_local.state_dict())

    agent = DQNAgentGoal(
        qnetwork_local=qnetwork_local,
        qnetwork_target=qnetwork_target,
        action_size=action_size,
        config=config
    )
    return agent
