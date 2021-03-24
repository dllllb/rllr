import torch
import numpy as np
from collections import deque
import random

import torch.nn as nn
import torch.nn.functional as F
import torch.optim


class ReplayBuffer:
    """
    Memory buffer for saving trajectories
    """

    def __init__(self, buffer_size, batch_size, device):
        self.buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.device = device

    def add(self, states, goal_states, rewards, next_states, dones):
        self.buffer.append((states, goal_states, rewards, next_states, dones))

    def sample(self):

        batch = random.sample(self.buffer, k=self.batch_size)

        def _vstack(arr):
            if arr and isinstance(arr[0], dict):
                arr = [x['image'] for x in arr]
            arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
            return torch.from_numpy(arr).float().to(self.device)

        states, goal_states, rewards, next_states, dones  = map(_vstack, zip(*batch))
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return states, goal_states, rewards, next_states, dones

    def is_enough(self):
        return len(self.buffer) > self.batch_size


class ActorNetwork(nn.Module):

    """
    Actor network for DDPG master agent.
    """

    def __init__(self, emb_size, state_encoder, hidden_size=64):
        super(ActorNetwork, self).__init__()
        self.state_encoder = state_encoder
        self.layer1 = nn.Linear(state_encoder.output_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, emb_size)

    def forward(self, states):
        x = self.state_encoder(states)
        x = F.relu(self.layer1(x))
        return torch.tanh(self.layer2(x))


class CriticNetwork(nn.Module):
    """
    Critic network for DDPG master agent
    """

    def __init__(self, emb_size, state_encoder, hidden_size=64):
        super(CriticNetwork, self).__init__()
        self.state_encoder = state_encoder
        self.layer1 = nn.Linear(state_encoder.output_size + emb_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, 1)

    def forward(self, states, goal_embs):
        x = self.state_encoder(states)
        x = torch.cat((x, goal_embs), 1)
        x = F.relu(self.layer1(x))
        return F.relu(self.layer2(x))


class MasterNetwork(nn.Module):
    """
    Actor-critic model for DDPG
    """

    def __init__(self, emb_size, state_encoder, learning_rate_critic=1e-4, learning_rate_actor=1e-4, tau=0.01, gamma=0.99):
        super(MasterNetwork, self).__init__()
        self.tau = tau
        self.gamma = gamma
        self.actor = ActorNetwork(emb_size, state_encoder)
        self.critic = CriticNetwork(emb_size, state_encoder)
        self.target_actor = ActorNetwork(emb_size, state_encoder)
        self.target_critic = CriticNetwork(emb_size, state_encoder)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_actor)

    def update_actor(self, states):
        self.actor_optimizer.zero_grad()
        goal_states = self.actor.forward(states)
        loss = -self.critic.forward(states, goal_states).mean()
        loss.backward()
        self.actor_optimizer.step()

    def update_critic(self, states, goal_states, rewards, dones, next_states):
        self.critic_optimizer.zero_grad()
        next_goal_states = self.target_actor.forward(next_states)
        q_target = self.target_critic.forward(next_states, next_goal_states).detach()
        y = rewards + self.gamma * (1 - dones) * q_target
        q = self.critic.forward(states, goal_states)
        loss = F.mse_loss(q, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
        self.critic_optimizer.step()

    def update_target_networks(self):
        critic_params = zip(self.target_critic.parameters(), self.critic.parameters())
        for target_param, param in critic_params:
            updated_params = self.tau * param.data + (1 - self.tau) * target_param.data
            target_param.data.copy_(updated_params)

        actor_params = zip(self.target_actor.parameters(), self.actor.parameters())
        for target_param, param in actor_params:
            updated_params = self.tau * param.data + (1 - self.tau) * target_param.data
            target_param.data.copy_(updated_params)


class DDPGAgentMaster:
    """ Deep deterministic policy gradient implementation for master agent.
        Master agent sets goal for worker to achieve higher future reward.
    """

    def __init__(self, master_network, config, buffer_size=int(1e6), batch_size=512,
                 noise_decay=0.995, min_noise=0.01, explore=True,
                 update_step=4, ):
        """Initializes an Agent.

        Params:
        -------
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """

        self.device = torch.device(config.agent['device'])
        self.master_network = master_network.to(self.device)
        self.replay_buffer = ReplayBuffer(buffer_size, batch_size, self.device)
        self.iter = 0
        self.update_step = update_step
        self.explore = explore
        self.act_noise = 1
        self.noise_decay = noise_decay
        self.min_noise = min_noise

    def _vstack(self, arr):
        if arr and isinstance(arr[0], dict):
            arr = [x['image'] for x in arr]
        arr = np.vstack([np.expand_dims(x, axis=0) for x in arr])
        return torch.from_numpy(arr).float().to(self.device)

    def sample_goal(self, state):

        goal_state = self.master_network.target_actor.forward(self._vstack([state]))

        if self.explore:
            goal_state += self.act_noise * torch.randn(goal_state.size()).to(self.device)

        return np.clip(goal_state.cpu().data.numpy(), -1, 1)[0]

    def update(self, states, goal_states, rewards, next_states, dones):

        self.replay_buffer.add(states, goal_states, rewards, next_states, dones)

        if (self.iter < self.update_step) or not self.replay_buffer.is_enough():
            self.iter += 1
            return

        self.iter = 0
        self._update_master()

    def reset_episode(self):
        self.act_noise = max(self.noise_decay * self.act_noise, self.min_noise)

    def _update_master(self):
        states, goal_states, rewards, next_states, dones = self.replay_buffer.sample()
        self.master_network.update_critic(states, goal_states, rewards, dones, next_states)
        self.master_network.update_actor(states)
        self.master_network.update_target_networks()