import torch
import torch.nn as nn
import gym

from learner import BufferedLearner, Updater
from pgrad import PGUpdater
from policy import Policy


class WorldModel:
    def update(self, next_state, reward: float):
        pass

    def end_episode(self):
        pass

    def __call__(self, state):
        """
        :param state: current state of the environment
        :param action: proposed action
        :return: state embedding and predicted reward
        """
        pass


def mlp_encoder(env: gym.Env, embed_size):
    state_space = env.observation_space.shape[0]

    encoder = nn.Sequential(
        nn.Linear(state_space, 128, bias=False),
        nn.Dropout(p=0.6),
        nn.ReLU(),
        nn.Linear(128, embed_size, bias=False)
    )

    return encoder


class MLPNextStatePred(nn.Module):
    def __init__(self, encoder, embed_size):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Linear(embed_size+1, embed_size, bias=False)

    def forward(self, state, action):
        s = torch.from_numpy(state).type(torch.FloatTensor)
        e = self.encoder(s)
        a = torch.tensor(action).unsqueeze(-1).type(torch.FloatTensor)
        sa = torch.cat([e, a])
        ns = self.predictor(sa)
        return ns


class MLPRewardPred(nn.Module):
    def __init__(self, encoder, embed_size):
        super().__init__()
        self.encoder = encoder
        self.predictor = nn.Linear(embed_size+1, 1, bias=False)

    def forward(self, state, action):
        s = torch.from_numpy(state).type(torch.FloatTensor)
        e = self.encoder(s)
        a = torch.tensor(action).unsqueeze(-1).type(torch.FloatTensor)
        sa = torch.cat([e, a])
        r = self.predictor(sa)
        return r


class MSEUpdater(Updater):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.mse = nn.MSELoss()

    def __call__(self, context, state_hist, reward_hist):
        prh = torch.stack(context)
        trh = torch.FloatTensor(reward_hist)
        loss = self.mse(prh, trh)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class DistUpdater(Updater):
    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __call__(self, context, state_hist, reward_hist):
        psh = torch.stack(context)
        tsh = torch.stack(state_hist)
        loss = torch.dist(psh, tsh).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class SPPolicy(BufferedLearner, Policy):
    def __init__(self, sp_nn, policy_nn, encoder, updater):
        super().__init__(updater)
        self.sp_nn = sp_nn
        self.policy_nn = policy_nn
        self.encoder = encoder

    def update(self, context, state, reward):
        s = torch.from_numpy(state).type(torch.FloatTensor)
        e = self.encoder(s)
        super().update(context, e, reward)

    def __call__(self, state):
        c = self.policy_nn(state)
        action = c.sample().item()
        next_state = self.sp_nn(state, action)
        return action, next_state


class RPPolicy(BufferedLearner, Policy):
    def __init__(self, rp_nn, policy_nn, updater):
        super().__init__(updater)
        self.rp_nn = rp_nn
        self.policy_nn = policy_nn

    def __call__(self, state):
        c = self.policy_nn(state)
        action = c.sample().item()
        reward = self.rp_nn(state, action)
        return action, reward
