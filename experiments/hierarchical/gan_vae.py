import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from utils import init_params, rollout
from env import gen_env_with_seed
from tqdm import trange
from rllr.env import make_vec_envs
from matplotlib import pyplot as plt


EMB_SIZE = 128


class VAE(nn.Module):
    def __init__(self, state_shape, emb_size=EMB_SIZE):
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

        self.mu = nn.Linear(in_features=emb_size, out_features=emb_size)
        self.std = nn.Linear(in_features=emb_size, out_features=emb_size)

        self.dec = nn.Sequential(
            nn.Linear(emb_size, 4096),
            nn.Unflatten(dim=1, unflattened_size=(16, 16, 16)), #16, 18, 18
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=c, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.apply(init_params)

    def encode(self, x):
        with torch.no_grad():
            hid = self.enc(x)
            return self.mu(hid)

    def decode(self, z):
        return self.dec(z)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        hid = self.enc(x.float())
        mu, logvar = self.mu(hid), self.std(hid)
        z = self.sample(mu, logvar)
        rx = self.dec(z)
        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size


class Discriminator(nn.Module):
    def __init__(self, state_shape, emb_size=EMB_SIZE):
        super(Discriminator, self).__init__()

        c, w, h = state_shape

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(4096, emb_size), #5184
        )

        self.fc = nn.Sequential(
            nn.Linear(emb_size, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 1),
        )

        self.apply(init_params)

    def forward(self, x):
        enc = self.enc(x)
        return self.fc(enc)


class GANVAE:
    def __init__(self, state_shape, batch_size=32, device='cpu'):
        self.vae = VAE(state_shape).to(device)
        self.vae_opt = torch.optim.RMSprop(self.vae.parameters(), lr=5e-5)

        self.discr = Discriminator(state_shape).to(device)
        self.discr_opt = torch.optim.RMSprop(self.discr.parameters(), lr=5e-5)

        self.batch_size = batch_size
        self.device = device

    def update(self, states):
        states = states.float() / 255.
        noize = torch.randn((states.shape[0], EMB_SIZE), device=self.device)

        gen_loss_epoch = 0
        discr_loss_epoch = 0
        vae_loss_epoch = 0
        n_updates = states.shape[0] // self.batch_size

        for i in range(n_updates):
            ids = torch.randint(0, states.shape[0], (self.batch_size,))
            x = states[ids]

            rx, mu, logvar = self.vae(x)
            vae_loss = self.vae.loss(rx, x, mu, logvar)
            self.vae_opt.zero_grad()
            vae_loss.backward()
            self.vae_opt.step()
            vae_loss_epoch += vae_loss.item()

            for _ in range(5):
                self.discr_opt.zero_grad()

                Gz = self.vae.decode(noize[ids])
                logits_fake = self.discr(Gz).mean()
                logits_real = self.discr(x).mean()

                discr_loss = (logits_fake - logits_real)
                discr_loss.backward()

                for p in self.discr.parameters():
                    p.data.clamp_(-0.01, 0.01)
                # torch.nn.utils.clip_grad_norm_(self.discr.parameters(), 0.5)

                self.discr_opt.step()
                discr_loss_epoch += discr_loss.item()


            self.vae_opt.zero_grad()
            Gz = self.vae.decode(noize[ids])
            logits_fake = -self.discr(Gz).mean()
            logits_fake.backward()
            # torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 0.5)
            self.vae_opt.step()
            gen_loss_epoch += logits_fake.item()

        return vae_loss_epoch / n_updates, gen_loss_epoch / n_updates, discr_loss_epoch / n_updates / 5

    def generate(self, batch_size):
        with torch.no_grad():
            return self.vae.decode(torch.randn((batch_size, EMB_SIZE), device=self.device))


if __name__ == '__main__':
    device = 'cpu'

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(env_id),
        16,
        device
    )

    gan = GANVAE(env.observation_space.shape, batch_size=32, device=device)

    for _ in trange(10):
        states = rollout(env)
        vae_loss, gen_loss, discr_loss = gan.update(states)
        print(f'vae_loss {vae_loss}, gen_loss {gen_loss}, discr_loss {discr_loss}')
        torch.save(gan, 'gan.p')

    gan = torch.load('gan.p', map_location='cpu')
    gan.device = device
    imgs = gan.generate(100).numpy()
    for img in imgs:
        plt.imshow(img.transpose(1, 2, 0))
        plt.show()

