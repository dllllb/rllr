import numpy as np
import torch
import torch.nn as nn
import gym
from torch.distributions import Categorical

from learner import Updater, BufferedLearner
from policy import Policy


class NNPolicy(BufferedLearner, Policy):
    def __init__(self, model: nn.Module, updater: Updater):
        super().__init__(updater)
        self.model = model

    def __call__(self, state):
        c = self.model(state)
        action = c.sample()
        self.pred_buffer.append(c.log_prob(action).view(1))
        return action.item()


class MLPPolicy(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()
        state_space = env.observation_space.shape[0]
        action_space = env.action_space.n

        self.model = nn.Sequential(
            nn.Linear(state_space, 128, bias=False),
            nn.Dropout(p=0.6),
            nn.ReLU(),
            nn.Linear(128, action_space, bias=False),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        state = torch.from_numpy(state).type(torch.FloatTensor)
        c = Categorical(self.model(state))
        return c


class PGUpdater(Updater):
    def __init__(self, optimizer, gamma: float):
        self.optimizer = optimizer
        self.gamma = gamma
        
    def __call__(self, pred_hist, reward_hist):
        # See https://github.com/pytorch/examples reinforcement_learning/reinforce.py
    
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in reward_hist[::-1]:
            R = r + self.gamma * R
            rewards.insert(0,R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        loss = []
        for log_prob, reward in zip(pred_hist, rewards):
            loss.append((-log_prob * reward).view(1))
        loss = torch.cat(loss).sum()

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

def train_loop(env, policy, n_eposodes, episode_len=1000):
    running_reward = 10
    for episode in range(n_eposodes):
        time, _ = play_episode(env, policy, episode_len=episode_len, render=False)
        
        # Used to determine when the environment is solved
        running_reward = (running_reward * 0.99) + (time * 0.01)
        
        if episode % 50 == 0:
            print(f'Episode {episode}\tLast length: {time:5d}\tAverage length: {running_reward:.2f}')
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward} and the last episode runs to {time} time steps!")
            break


def play_episode(env, policy, render, episode_len=1000):
    state = env.reset()
    frames = []

    time = 0
    for i in range(episode_len):
        time = i

        if render:
            frames.append(env.render(mode='rgb_array'))

        action = policy(state)
        state, reward, done, _ = env.step(action)
        policy.update(reward)

        if done:
            break

    policy.end_episode()
    
    return time, frames


def test_train_loop():
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)

    nn = MLPPolicy(env)
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
    updater = PGUpdater(optimizer, gamma=.99)
    policy = NNPolicy(MLPPolicy(env), updater)

    train_loop(env=env, policy=policy, n_eposodes=1, episode_len=5)
