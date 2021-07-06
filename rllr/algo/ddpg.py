import torch
import logging

import torch.nn.functional as F
import torch.optim

from ..buffer import ReplayBuffer
from ..utils import convert_to_torch
from .core import Algo


logger = logging.getLogger(__name__)


class DDPG(Algo):
    """ Deep deterministic policy gradient implementation for master agent.
        Master agent sets goal for worker to achieve higher future reward.
    """

    def __init__(self,
                 actor_critic: torch.nn.Module,
                 device: torch.device,
                 explore: bool = True,
                 buffer_size: int = int(1e6),
                 batch_size: int = 128,
                 tau: float = 0.001,
                 gamma: float = 0.99,
                 update_step: int = 50,
                 start_noise: float = 1.,
                 noise_decay: float = 0.995,
                 min_noise: float = 0.01,
                 epochs: int = 50,
                 steps_per_epoch: int = 1,
                 policy_delay: int = 1,
                 learning_rate_critic: float = 1e-3,
                 learning_rate_actor: float = 1e-3,
                 actor_grad_clipping: int = None,
                 critic_grad_clipping: int = 1,
                 action_range: tuple = (-1., 1.)):

        self.device = device
        self.actor_critic = actor_critic.to(self.device)
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
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
        self.action_range = action_range

    def act(self, state):
        action = self.actor_critic.actor.forward(convert_to_torch([state], device=self.device))

        if self.explore:
            action += self.act_noise * torch.randn(action.size()).to(self.device)
        action = action.clamp(-1, 1).cpu().data.numpy()[0]
        return self.unscale_action(action)

    def scale_action(self, actions):
        low = self.action_range[0]
        high = self.action_range[1]
        return (actions - low) / (high - low) * 2. - 1.

    def unscale_action(self, scaled_actions):
        low = self.action_range[0]
        high = self.action_range[1]
        return low + (scaled_actions + 1.)/2. * (high - low)

    def update(self, states, actions, rewards, next_states, dones):

        scaled_actions = self.scale_action(actions)
        self.replay_buffer.add(states, scaled_actions, rewards, next_states, dones)

        if (self.iter < self.update_step) or not self.replay_buffer.is_enough():
            self.iter += 1
            return

        self.iter = 0
        for _ in range(self.epochs):
            self._update_actor_critic()

    def reset_episode(self):
        self.act_noise = max(self.noise_decay * self.act_noise, self.min_noise)

    def _update_actor_critic(self):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample()
        n_updates = 0
        for _ in range(self.steps_per_epoch):
            n_updates += 1
            self._update_critic(states, actions, rewards, dones, next_states)
            if n_updates % self.policy_delay == 0:
                self._update_actor(states)
                self._update_target_networks()

    def _update_actor(self, states):
        actions = self.actor_critic.actor.forward(states)
        loss = -self.actor_critic.critic.forward(states, actions).mean()
        self.actor_optimizer.zero_grad()
        loss.backward()
        if self.actor_grad_clipping:
            torch.nn.utils.clip_grad_norm_(self.actor_critic.actor.parameters(), self.actor_grad_clipping)
        self.actor_optimizer.step()

    def _update_critic(self, states, actions, rewards, dones, next_states):
        next_actions = self.actor_critic.target_actor.forward(next_states)
        q_target = self.actor_critic.target_critic.forward(next_states, next_actions).detach()
        y = rewards + self.gamma * (1 - dones) * q_target
        q = self.actor_critic.critic.forward(states, actions)
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