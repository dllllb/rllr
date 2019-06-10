import torch
import torch.nn as nn
import gym
import numpy as np

from learner import BufferedLearner, Updater


class WorldModel:
    def update(self, next_state, reward: float):
        pass

    def end_episode(self):
        pass

    def __call__(self, state, action):
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


class WMUpdater(Updater):
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.mse = nn.MSELoss()

    def __call__(self, pred_hist, true_hist):
        state_pred_hist, reward_pred_hist = zip(*pred_hist)
        state_hist, reward_hist = zip(*true_hist)
        # Calculate loss
        psh = torch.stack(state_pred_hist)
        tsh = torch.stack(state_hist)
        state_loss = torch.dist(psh, tsh).mean()

        prh = torch.FloatTensor(reward_pred_hist)
        trh = torch.FloatTensor(reward_hist)
        rew_loss = self.mse(prh, trh)

        loss = state_loss + rew_loss

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


class NNWorldModel(BufferedLearner, WorldModel):
    def __init__(self, sp_nn, rp_nn, encoder, updater):
        super().__init__(updater)
        self.sp_nn = sp_nn
        self.rp_nn = rp_nn
        self.encoder = encoder

    def update(self, state, action):
        s = torch.from_numpy(state).type(torch.FloatTensor)
        super().update(self.encoder(s), action)

    def __call__(self, state, action):
        next_state = self.sp_nn(state, action)
        reward = self.rp_nn(state, action)
        self.pred_buffer.append((next_state, reward))
        return next_state, reward
