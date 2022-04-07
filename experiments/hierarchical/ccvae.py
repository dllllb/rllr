import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from utils import init_params, rollout
from env import gen_env_with_seed
from tqdm import trange
from rllr.env import make_vec_envs
from matplotlib import pyplot as plt


EMB_SIZE = 2
EMB0_SIZE = 128


class VAE(nn.Module):
    def __init__(self, state_shape, emb_size=EMB_SIZE, emb0_size=EMB0_SIZE):
        super(VAE, self).__init__()
        self.state_shape = state_shape
        self.beta = 1.0

        c, w, h = state_shape

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(4096, emb_size), #5184
        )

        self.enc0 = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(4096, emb0_size), #5184
        )

        self.mu = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.std = nn.Linear(in_features=emb_size, out_features=emb_size)

        self.dec = nn.Sequential(
            nn.Linear(emb_size + emb0_size, 4096),
            nn.Unflatten(dim=1, unflattened_size=(16, 16, 16)), #16, 18, 18
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=c, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.dec0 = nn.Sequential(
            nn.Linear(emb0_size, 4096),
            nn.Unflatten(dim=1, unflattened_size=(16, 16, 16)),  # 16, 18, 18
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=c, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.apply(init_params)

    def encode(self, x, x0):
        with torch.no_grad():
            hid = self.enc(x)
            z = self.mu(hid)

            z0 = self.enc0(x0)
            z = torch.cat([z, z0], dim=1)
            return z

    def decode(self, z):
        return self.dec(z)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def forward(self, x, x0):
        z0 = self.enc0(x0)
        rx0 = self.dec0(z0)

        hid = self.enc(x)
        mu, logvar = self.mu(hid), self.std(hid)
        z = self.sample(mu, logvar)
        z = torch.cat([z, z0], dim=1)
        rx = self.dec(z)
        return rx0, rx, mu, logvar

    def loss(self, recon_x0, x0, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')
        recon_loss0 = F.binary_cross_entropy(recon_x0, x0, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge + recon_loss0) / x.shape[0]  # divide total loss by batch size


class CCVAE:
    def __init__(self, state_shape, batch_size=32, device='cpu'):
        self.vae = VAE(state_shape).to(device)
        self.vae_opt = torch.optim.RMSprop(self.vae.parameters(), lr=5e-5)

        self.batch_size = batch_size
        self.device = device

    def update(self, states, x0):
        states = states.float() / 255.
        x0 = x0.float() / 255.

        vae_loss_epoch = 0
        n_updates = states.shape[0] // self.batch_size

        for i in range(n_updates):
            ids = torch.randint(0, states.shape[0], (self.batch_size,))
            x = states[ids]

            rx0, rx, mu, logvar = self.vae(x, x0)
            vae_loss = self.vae.loss(rx0, x0, rx, x, mu, logvar)
            self.vae_opt.zero_grad()
            vae_loss.backward()
            self.vae_opt.step()
            vae_loss_epoch += vae_loss.item()

        return vae_loss_epoch / n_updates

    def generate(self, batch_size, x0):
        with torch.no_grad():
            hid = torch.randn((batch_size, EMB_SIZE), device=self.device)
            hid0 = self.vae.enc0(x0)
            hid = torch.cat([hid, hid0], dim=1)
            return self.vae.decode(hid)


if __name__ == '__main__':
    device = 'cpu'

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(env_id),
        16,
        device
    )

    x0 = env.reset()[0].repeat(32, 1, 1, 1)

    gan = CCVAE(env.observation_space.shape, batch_size=32, device=device)

    for _ in trange(1000):
        states = rollout(env)
        vae_loss = gan.update(states, x0)
        print(f'vae_loss {vae_loss}')
        torch.save(gan, 'gan.p')

    gan = torch.load('gan.p', map_location='cpu')
    gan.device = device
    states = rollout(env) / 255.
    x0 = env.reset()[0].repeat(states.size(0), 1, 1, 1).float() / 255.
    with torch.no_grad():
        hid = gan.vae.enc(states)
        hid0 = gan.vae.enc0(x0)
        hid = torch.cat([hid, hid0], dim=1)
        reco = gan.vae.decode(hid).numpy()
    imgs = gan.generate(states.size(0), x0).numpy()
    states = states.numpy()
    for rec, state, img in zip(reco, states, imgs):
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(state.transpose(1, 2, 0))
        ax[1].imshow(rec.transpose(1, 2, 0))
        ax[2].imshow(img.transpose(1, 2, 0))
        plt.show()

