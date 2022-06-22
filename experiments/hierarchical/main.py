import torch
from torch import nn
import os
from tqdm import trange
import numpy as np
from env import gen_env_with_seed
from vae import VAE
from utils import train_vae, rollout
from matplotlib import pyplot as plt

from rllr.env import make_vec_envs


class Master(nn.Module):
    def __init__(self, vae, emb_size=256):
        super(Master, self).__init__()
        self.enc = vae.enc
        self.dec = vae.dec
        for param in self.enc.parameters():
            param.requires_grad = False
        for param in self.dec.parameters():
            param.requires_grad = False

        self.mu = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=emb_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )

        self.std = nn.Sequential(
            nn.Linear(in_features=emb_size, out_features=emb_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=emb_size, out_features=emb_size)
        )

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def forward(self, t):
        hid = self.enc(x.float() / 255.)
        mu, logvar = self.mu(hid), self.std(hid)
        z = self.sample(mu, logvar)
        rx = self.dec(z)
        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x.float() / 255., reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size


def update_master(vae, optimizer, states, initial_states, batch_size=32):
    rec_loss = []
    for _ in range(states.shape[0] // batch_size):
        ids = torch.randint(0, states.shape[0], (batch_size,))
        imgs = states[ids]
        rec, mu, logvar = vae(imgs)

        optimizer.zero_grad()
        loss = vae.loss(rec, initial_states, mu, logvar)
        loss.backward()
        optimizer.step()

        rec_loss.append(loss.detach().cpu().numpy())
    return np.mean(rec_loss)


if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(env_id),
        num_processes=32,
        device=device
    )

    vae = VAE(env.observation_space.shape, 256).to(device)

    if os.path.isfile('vae.pt'):
        vae.load_state_dict(torch.load('vae.pt', map_location=device))
    else:
        train_vae(env, vae)

    master = Master(vae).to(device)

    optimizer = torch.optim.Adam(vae.parameters())

    initial_states = env.reset()
    for epoch in trange(1000):
        target_states = rollout(env)
        rec_loss = update_master(vae, optimizer, target_states, initial_states, batch_size=32)
        print('rec_loss', rec_loss)
        torch.save(master.state_dict(), 'master.pt')

