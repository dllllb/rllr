import torch
import logging

from typing import Type

import torch.nn.functional as F
import torch.optim

from ..buffer import ReplayBuffer
from ..utils import convert_to_torch


logger = logging.getLogger(__name__)


class DDPG:
    """ Deep deterministic policy gradient implementation for master agent.
        Master agent sets goal for worker to achieve higher future reward.
    """

    def __init__(self,
                 actor_critic: torch.nn.Module,
                 replay_buffer: ReplayBuffer,
                 device: torch.device,
                 explore: bool = True,
                 tau: float = 0.001,
                 gamma: float = 0.99,
                 update_step: int = 50,
                 start_noise: float = 0.1,
                 noise_decay: float = 1,
                 min_noise: float = 0.1,
                 epochs: int = 50,
                 steps_per_epoch: int = 1,
                 policy_delay: int = 1,
                 learning_rate_critic: float = 1e-3,
                 learning_rate_actor: float = 1e-3,
                 actor_grad_clipping: int = None,
                 critic_grad_clipping: int = None):

        self.device = device
        self.actor_critic = actor_critic.to(self.device)
        self.buffer_size = replay_buffer.buffer_size
        self.batch_size = replay_buffer.batch_size
        self.replay_buffer = replay_buffer
        self.iter = 0
        self.update_step = update_step
        self.explore = explore
        self.act_noise = start_noise
        self.noise_decay = noise_decay
        self.min_noise = min_noise
        self.epochs = epochs
        self.steps_per_epoch = steps_per_epoch
        self.policy_delay = policy_delay
        self.tau = tau
        self.gamma = gamma
        self.critic_optimizer = torch.optim.Adam(self.actor_critic.critic.parameters(), lr=learning_rate_critic)
        self.actor_optimizer = torch.optim.Adam(self.actor_critic.actor.parameters(), learning_rate_actor)
        self.actor_grad_clipping = actor_grad_clipping
        self.critic_grad_clipping = critic_grad_clipping

    def act(self, state):
        goal_state = self.actor_critic.target_actor.forward(convert_to_torch([state], device=self.device))

        if self.explore:
            goal_state += self.act_noise * torch.randn(goal_state.size()).to(self.device)
        return goal_state.clamp(-1, 1).cpu().data.numpy()[0]

    def update(self, states, goal_states, rewards, next_states, dones):

        self.replay_buffer.add(states, goal_states, rewards, next_states, dones)

        if (self.iter < self.update_step) or not self.replay_buffer.is_enough():
            self.iter += 1
            return

        self.iter = 0
        for _ in range(self.epochs):
            self._update_actor_critic()

    def reset_episode(self):
        self.act_noise = max(self.noise_decay * self.act_noise, self.min_noise)

    def _update_actor_critic(self):
        states, goal_states, rewards, next_states, dones = self.replay_buffer.sample()
        n_updates = 0
        for _ in range(self.steps_per_epoch):
            n_updates += 1
            self._update_critic(states, goal_states, rewards, dones, next_states)
            if n_updates % self.policy_delay == 0:
                self._update_actor(states)
                self._update_target_networks()

    def _update_actor(self, states):
        goal_states = self.actor_critic.actor.forward(states)
        loss = -self.actor_critic.critic.forward(states, goal_states).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.actor_grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.actor_grad_clipping)
        self.actor_optimizer.step()

    def _update_critic(self, states, goal_states, rewards, dones, next_states):
        next_goal_states = self.actor_critic.target_actor.forward(next_states)
        q_target = self.actor_critic.target_critic.forward(next_states, next_goal_states).detach()
        y = rewards + self.gamma * (1 - dones) * q_target
        q = self.actor_critic.critic.forward(states, goal_states)
        loss = F.mse_loss(q, y)
        self.critic_optimizer.zero_grad()
        loss.backward()
        if self.critic_grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.critic.parameters(), self.critic_grad_clipping)
        self.critic_optimizer.step()

    def _update_target_networks(self):
        critic_params = zip(self.actor_critic.target_critic.parameters(), self.actor_critic.critic.parameters())
        for target_param, param in critic_params:
            updated_params = self.tau * param.data + (1 - self.tau) * target_param.data
            target_param.data.copy_(updated_params)

        actor_params = zip(self.actor_critic.target_actor.parameters(), self.actor_critic.actor.parameters())
        for target_param, param in actor_params:
            updated_params = self.tau * param.data + (1 - self.tau) * target_param.data
            target_param.data.copy_(updated_params)


def get_ddpg_agent(master_network, conf):
    device = torch.device(conf['device'])
    replay_buffer = ReplayBuffer(conf['buffer_size'], conf['batch_size'], device)
    return DDPG(master_network, replay_buffer, device, explore=conf['explore'], update_step=conf['update_step'],
                start_noise=conf['start_noise'], noise_decay=conf['noise_decay'], min_noise=conf['min_noise'])