import numpy as np
from tqdm import trange, tqdm
import torch
from matplotlib import pyplot as plt
from torch import nn
import gym_minigrid


def init_params(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0)


def rollout(env):
    obs, done, info = env.reset(), False, []

    states = []
    for _ in range(128):
        states.append(obs)
        action = torch.randint(0, env.action_space.n, (obs.size(0), 1))
        obs, reward, done, info = env.step(action)
    states.append(obs)

    return torch.stack(states, dim=0).transpose(1, 0).reshape(-1, *env.observation_space.shape)


def update_vae(vae, optimizer, states, batch_size=32):
    rec_loss = []
    for _ in range(states.shape[0] // batch_size):
        ids = torch.randint(0, states.shape[0], (batch_size,))
        imgs = states[ids]
        rec, mu, logvar = vae(imgs)

        optimizer.zero_grad()
        loss = vae.loss(rec, imgs, mu, logvar)
        loss.backward()
        optimizer.step()

        rec_loss.append(loss.detach().cpu().numpy())
    return np.mean(rec_loss)


def train_vae(env, vae, n_epoch=1000, batch_size=32):
    optimizer = torch.optim.Adam(vae.parameters())

    for epoch in trange(n_epoch):
        states = rollout(env)
        rec_loss = update_vae(vae, optimizer, states, batch_size=batch_size)
        print('rec_loss', rec_loss)
        torch.save(vae.state_dict(), 'vae.pt')

