import numpy as np
import torch
import torch.nn as nn
import gym
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.metrics import RunningAverage
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
        return action.item(), c.log_prob(action).view(1)


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


class ConvPolicy(nn.Module):
    def __init__(self, env: gym.Env):
        super().__init__()

        img_height, img_width, img_channels = env.observation_space.shape

        self.conv = nn.Sequential(
            nn.Conv2d(img_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        o = self.conv(torch.zeros(1, img_channels, img_height, img_width))
        conv_out_size = int(np.prod(o.size()))

        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, env.action_space.n),
            nn.Softmax(dim=-1)
        )

    def forward(self, state):
        state = torch.from_numpy(state).float() / 256
        if len(state.shape) == 3:
            state = state.view(1, *state.shape)
            state = state.permute(0, 3, 1, 2)
        elif len(state.shape) == 2:
            state = state.view(1, 1, *state.shape)

        conv_out = self.conv(state).view(state.size(0), -1)
        c = Categorical(self.fc(conv_out))
        return c


class PGUpdater(Updater):
    def __init__(self, optimizer, gamma: float):
        self.optimizer = optimizer
        self.gamma = gamma
        
    def __call__(self, context_hist, state_hist, reward_hist):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in reward_hist[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        loss = []
        for log_prob, reward in zip(context_hist, rewards):
            loss.append((-log_prob * reward).view(1))
        loss = torch.cat(loss).sum()

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

def train_loop(env, policy, n_episodes, episode_len=1000, render=False):
    def train_timestep(engine, timestep):
        state = engine.state.observation
        action, context = policy(state)
        state, reward, done, _ = env.step(action)
        policy.update(context, state, reward)
        engine.state.total_reward += reward
        if done:
            engine.terminate_epoch()
            engine.state.timestep = timestep

        return reward

    trainer = Engine(train_timestep)

    @trainer.on(Events.EPOCH_STARTED)
    def reset_environment_state(engine):
        engine.state.total_reward = 0
        engine.state.observation = env.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_model(engine):
        policy.end_episode()
        print(f'session reward: {engine.state.total_reward}')

    if render:
        @trainer.on(Events.ITERATION_COMPLETED)
        def render(_):
            env.render()

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer)

    timesteps = list(range(episode_len))
    trainer.run(timesteps, max_epochs=n_episodes)


def test_train_loop():
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)

    nn = MLPPolicy(env)
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
    updater = PGUpdater(optimizer, gamma=.99)
    policy = NNPolicy(nn, updater)

    train_loop(env=env, policy=policy, n_episodes=1, episode_len=5)


def test_conv_policy():
    env = gym.make('BreakoutDeterministic-v4')
    env.seed(1)
    torch.manual_seed(1)

    nn = ConvPolicy(env)
    optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
    updater = PGUpdater(optimizer, gamma=.99)
    policy = NNPolicy(nn, updater)

    train_loop(env=env, policy=policy, n_episodes=1, episode_len=5)
