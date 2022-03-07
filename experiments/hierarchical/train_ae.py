from pyhocon import ConfigFactory
from rllr.env.vec_wrappers import make_vec_envs
import torch
import numpy as np
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import AdamW
from matplotlib import pyplot as plt
from tqdm import trange, tqdm


def rollout(env):
    obs, done, info = env.reset(), False, []

    states = []
    while not done:
        states.append(obs)
        action = torch.randint(0, 3, obs.size(0))
        obs, reward, done, info = env.step(action)
    states.append(obs)

    return torch.cat(states, dim=0)


class VAE(nn.Module):
    def __init__(self, state_shape):
        super(VAE, self).__init__()
        self.state_shape = state_shape
        self.output_size = 256
        self.beta = 1.0

        c, w, h = state_shape

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=32, kernel_size=8, stride=4),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.LeakyReLU(inplace=True),

            nn.Flatten(),
            nn.Linear(in_features=3136, out_features=256)
        )

        self.mu = nn.Linear(in_features=256, out_features=256)
        self.std = nn.Linear(in_features=256, out_features=256)

        self.dec = nn.Sequential(
            nn.Linear(256, 3136),
            nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=3),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=c, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        hid = self.enc(x)
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


def train_vae(env):
    n_epoch = 1000
    batch_size = 32
    opt = AdamW(aenc.parameters())

    for epoch in trange(n_epoch):
        rec_loss = []
        states = rollout(env)
        for _ in range(states.shape[0] // batch_size):
            ids = torch.randint(0, states.shape[0], (batch_size,))
            imgs = (states[ids] / 255.).to(device)
            rec, mu, logvar = aenc(imgs)

            opt.zero_grad()
            loss = aenc.loss(rec, imgs, mu, logvar)
            loss.backward()
            opt.step()

            rec_loss.append(loss.detach().cpu().numpy())
        print(np.mean(rec_loss))
        torch.save(aenc.state_dict(), 'aenc.p')


def test_vae(env):
    states = rollout(env)
    ids = torch.randint(0, states.shape[0] - 1, (states.shape[0],))
    for i in tqdm(ids):
        img = states[i].unsqueeze(dim=0) / 255.

        with torch.no_grad():
            rec, _, _ = aenc(img)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img[0].permute(1, 2, 0), cmap='gray')
        axarr[1].imshow(rec[0].permute(1, 2, 0), cmap='gray')
        plt.show()


if __name__ == '__main__':
    device = 'cpu'
    config = ConfigFactory.parse_file('conf/montezuma_rnd_ppo.hocon')

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, 0, render=False),
        num_processes=1,
        device=device
    )

    aenc = VAE(env.observation_space.shape).to(device)
    train_vae(env)

    aenc.load_state_dict(torch.load(open('aenc.p', 'rb'), map_location=device))
    test_vae(env)
