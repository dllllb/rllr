import torch
import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch.optim

from models import MasterNetwork
from replay import ReplayBuffer

from utils import convert_to_torch
import logging

logger = logging.getLogger(__name__)


class CriticNetwork(nn.Module):
    """
    Critic network for DDPG master agent
    """

    def __init__(self, emb_size, state_encoder, hidden_size=64):
        super(CriticNetwork, self).__init__()
        self.state_encoder = state_encoder
        self.layer = nn.Linear(state_encoder.output_size + emb_size, 1)

    def forward(self, states, goal_embs):
        x = self.state_encoder(states)
        x = torch.cat((x, goal_embs), 1)
        return F.relu(self.layer(x))


class MasterCriticNetwork(nn.Module):
    """
    Actor-critic model for DDPG
    """

    def __init__(self, emb_size, state_encoder, goals_state_encoder, config,
                 learning_rate_critic=1e-3, learning_rate_actor=1e-3, tau=0.001, gamma=0.99,
                 actor_grad_clipping=False, critic_grad_clipping=False):
        super(MasterCriticNetwork, self).__init__()
        self.tau = tau
        self.gamma = gamma
        self.actor = MasterNetwork(emb_size, goals_state_encoder, config)
        self.critic = CriticNetwork(emb_size, state_encoder)
        self.target_actor = MasterNetwork(emb_size, goals_state_encoder, config)
        self.target_critic = CriticNetwork(emb_size, state_encoder)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=learning_rate_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=learning_rate_actor)
        self.actor_grad_clipping = actor_grad_clipping
        self.critic_grad_clipping = critic_grad_clipping

    def update_actor(self, states):
        goal_states = self.actor.forward(states)
        loss = -self.critic.forward(states, goal_states).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.actor_grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clipping)
        self.actor_optimizer.step()

    def update_critic(self, states, goal_states, rewards, dones, next_states, act_noise):
        noise = act_noise * torch.randn(goal_states.size()).to(act_noise.device)
        next_goal_states = (self.target_actor.forward(next_states) + noise).clamp(-1, 1)
        q_target = self.target_critic.forward(next_states, next_goal_states).detach()
        y = rewards + self.gamma * (1 - dones) * q_target
        q = self.critic.forward(states, goal_states)
        loss = F.mse_loss(q, y)
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.critic_grad_clipping)
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

    def __init__(self, master_network, conf):
        """
        Initializes an Agent.
        """

        self.device = torch.device(conf['device'])
        self.master_network = master_network.to(self.device)
        self.replay_buffer = ReplayBuffer(conf['buffer_size'], conf['batch_size'], self.device)
        self.iter = 0
        self.update_step = conf['update_step']
        self.explore = conf['explore']
        self.act_noise = conf.get('start_noise', 1)
        self.noise_decay = conf['noise_decay']
        self.min_noise = conf['min_noise']
        self.epochs = conf.get('epochs', 1)
        self.steps_per_epoch = conf.get('steps_per_epochs', 1)
        self.policy_delay = conf.get('policy_delay', 2)

    def sample_goal(self, state):
        goal_state = self.master_network.target_actor.forward(convert_to_torch([state], device=self.device))

        if self.explore:
            goal_state += self.act_noise * torch.randn(goal_state.size()).to(self.device)
        return goal_state.clamp(-1, 1).cpu().data.numpy()[0]

    def update(self, states, goal_states, rewards, next_states, dones):

        self.replay_buffer.add(states, goal_states, rewards, next_states, dones, self.act_noise)

        if (self.iter < self.update_step) or not self.replay_buffer.is_enough():
            self.iter += 1
            return

        self.iter = 0
        for _ in range(self.epochs):
            self._update_master()

    def reset_episode(self):
        self.act_noise = max(self.noise_decay * self.act_noise, self.min_noise)

    def _update_master(self):
        states, goal_states, rewards, next_states, dones, act_noise = self.replay_buffer.sample()
        n_updates = 0
        for _ in range(self.steps_per_epoch):
            n_updates += 1
            self.master_network.update_critic(states, goal_states, rewards, dones, next_states, act_noise)
            if n_updates % self.policy_delay == 0:
                self.master_network.update_actor(states)
                self.master_network.update_target_networks()
