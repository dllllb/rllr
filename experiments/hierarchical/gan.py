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


class Generator(nn.Module):
    def __init__(self, state_shape, emb_size=EMB_SIZE):
        super(Generator, self).__init__()

        c, w, h = state_shape

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

    def forward(self, noize):
        return self.dec(noize)


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


class GAN:
    def __init__(self, state_shape, batch_size=32, device='cpu'):
        self.gen = Generator(state_shape).to(device)
        self.gen_opt = torch.optim.RMSprop(self.gen.parameters(), lr=5e-5)

        self.discr = Discriminator(state_shape).to(device)
        self.discr_opt = torch.optim.RMSprop(self.discr.parameters(), lr=5e-5)

        self.batch_size = batch_size
        self.device = device

    def update(self, states):
        states = states.float() / 255.
        noize = torch.randn((states.shape[0], EMB_SIZE), device=self.device)

        gen_loss_epoch = 0
        discr_loss_epoch = 0
        n_updates = states.shape[0] // self.batch_size

        for i in range(n_updates):
            ids = torch.randint(0, states.shape[0], (self.batch_size,))
            x = states[ids]

            for _ in range(5):
                self.discr_opt.zero_grad()

                Gz = self.gen(noize[ids])
                logits_fake = self.discr(Gz).mean()
                logits_real = self.discr(x).mean()

                discr_loss = (logits_fake - logits_real)
                discr_loss.backward()

                for p in self.discr.parameters():
                    p.data.clamp_(-0.01, 0.01)
                # torch.nn.utils.clip_grad_norm_(self.discr.parameters(), 0.5)

                self.discr_opt.step()
                discr_loss_epoch += discr_loss.item()


            self.gen_opt.zero_grad()
            Gz = self.gen(noize[ids])
            logits_fake = -self.discr(Gz).mean()
            logits_fake.backward()
            # torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 0.5)
            self.gen_opt.step()
            gen_loss_epoch += logits_fake.item()


        return gen_loss_epoch / n_updates, discr_loss_epoch / n_updates / 5

    def generate(self, batch_size):
        with torch.no_grad():
            return self.gen(torch.randn((batch_size, EMB_SIZE), device=self.device))



if __name__ == '__main__':
    device = 'cpu'


    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(env_id),
        16,
        device
    )

    gan = GAN(env.observation_space.shape, batch_size=32, device=device)

    for _ in trange(0):
        states = rollout(env)
        gen_loss, discr_loss = gan.update(states)
        print(f'gen_loss {gen_loss}, discr_loss {discr_loss}')
        torch.save(gan, 'gan.p')

    gan = torch.load('gan.p', map_location='cpu')
    gan.device = device
    imgs = gan.generate(100).numpy()
    for img in imgs:
        plt.imshow(img.transpose(1, 2, 0))
        plt.show()






