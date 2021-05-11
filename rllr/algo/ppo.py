import torch
import torch.nn as nn
import numpy as np
from rllr.buffer import ReplayBuffer
from rllr.algo.core import Algo
from rllr.models.ppo import ActorCritic
from rllr.utils import convert_to_torch


class PPO(Algo):

    def __init__(self,
                 state_size,
                 action_size,
                 device,
                 T=128,
                 K_epochs=10,
                 lr=5e-3,
                 lamb=0.8,
                 gamma=0.97,
                 eps=0.2,
                 c1=0.5,
                 c2=0.01):
        """Initializes agent object

        Args:
         action_size - action space dimensions
         state_size - state space dimensions
         actor_critic - pretrained actor-critic network
         T - time steps to collect before agent updating
         K_epochs - number of steps while optimizing networcs
         lr - learning rate for Adam optimizer
         lamb - smoothing parameter for generalized advantage estimator
         gamma - decay
         eps - clipping threshold
         c1 - weight for critic loss
         c2 - weight for entropy loss

        """
        self.policy = ActorCritic(state_size, action_size).to(device)
        self.policy_old = ActorCritic(state_size, action_size).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory = ReplayBuffer(buffer_size=T, batch_size=T, device=device)
        self.T = T
        self.K_epochs = K_epochs
        self.c1 = c1
        self.c2 = c2
        self.lamb = lamb
        self.gamma = gamma
        self.epsilon = eps
        self.mse_loss = nn.MSELoss()
        self.device = device
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)

    def act(self, states):
        """Takes actions given batch of states

        Args:
         states - a batch of states

        Returns:
         actions - a batch of actions generated given states
        """
        states = convert_to_torch(states)
        with torch.no_grad():
            actions = self.policy_old.act(states)
            return actions.detach().cpu().numpy()

    def update(self, states, actions, rewards, next_states, dones):
        """Updates actor critic network
        """

        # Add to memory untill collect trajectories of length memory_size
        if not self.memory.is_enough():
            self.memory.add(states, actions, rewards, next_states, dones)
            return

        # Optimize
        for _ in range(self.K_epochs):
            loss = self._compute_loss()
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.memory.clear()

    def _compute_loss(self):

        # Iterate over actors and create batch
        loss = 0

        states, actions, rewards, next_states, dones = self.memory.get()
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards_to_go = self._compute_rewards_to_go(rewards, dones)

        values, logprobs, S = self.policy.evaluate(states, actions)

        with torch.no_grad():
            values_old, logprobs_old, _ = self.policy_old.evaluate(states, actions)
            values_next_old, _, _ = self.policy_old.evaluate(next_states, None)
            values_old = values_old.detach()
            values_next_old = values_next_old.detach()

        ratios = torch.exp(logprobs - logprobs_old.detach())
        advantages = self._compute_advantages(rewards, values_old, values_next_old, dones)

        # Compute surrogate loss with clipping
        s1 = ratios * advantages
        s2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages

        L_clip = torch.min(s1, s2)

        # Compute MSE loss for value functions
        L_vf = self.mse_loss(values, rewards_to_go)

        # Combine losses
        loss += -L_clip.mean() + self.c1 * L_vf - self.c2 * S.mean()

        return loss

    def _compute_advantages(self, rewards, values, next_values, dones):
        td_errors = rewards + self.gamma * next_values * (1 - dones) - values
        A, advantages = 0, []
        for t in reversed(range(len(td_errors))):
            A = td_errors[t] + (self.lamb * self.gamma) * A * (1 - dones[t])
            advantages.insert(0, A)
        return convert_to_torch(np.array(advantages), self.device)

    def _compute_rewards_to_go(self, rewards, dones):
        rewards_to_go = []
        R = 0
        for reward, done in zip(reversed(rewards), reversed(dones)):
            R = reward + self.gamma * R * (1 - done)
            rewards_to_go.insert(0, R)
        return convert_to_torch(np.array(rewards_to_go), self.device)

    def reset_episode(self):
        return True