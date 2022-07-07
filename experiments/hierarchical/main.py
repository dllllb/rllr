import torch
from torch import nn
from env import gen_env_with_seed
from utils import train_vae, rollout
from vae import VAE
from rllr.env import make_vec_envs
import os
from tqdm import trange


class Discriminator(nn.Module):
    def __init__(self, emb_size):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_size, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, 1)
        )

    def forward(self, t):
        return self.net(t)


class Master(nn.Module):
    def __init__(self, emb_size):
        super(Master, self).__init__()
        print(emb_size)
        self.mu = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(emb_size, emb_size)
        )

        self.logstd = nn.Sequential(
            nn.Linear(emb_size, emb_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(emb_size, emb_size)
        )

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def forward(self, t):
        mu, logstd = self.mu(t), self.logstd(t)
        return self.sample(mu, logstd)



class GAN:
    def __init__(self, state_shape, batch_size=32, device='cpu'):
        self.gen = Master(state_shape).to(device)
        self.gen_opt = torch.optim.RMSprop(self.gen.parameters(), lr=5e-5)

        self.discr = Discriminator(state_shape).to(device)
        self.discr_opt = torch.optim.RMSprop(self.discr.parameters(), lr=5e-5)

        self.batch_size = batch_size
        self.device = device

    def update(self, states):
        gen_loss_epoch = 0
        gen_dist = 0
        discr_loss_epoch = 0

        n_updates = states.shape[0] // self.batch_size

        for i in range(n_updates):
            ids = torch.randint(0, states.shape[0], (self.batch_size,))
            x = states[ids]

            for _ in range(5):
                self.discr_opt.zero_grad()

                Gz = self.gen(states[ids])
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
            Gz = self.gen(states[ids])
            mse_dist = torch.nn.functional.mse_loss(Gz, states[ids])
            gen_dist += mse_dist.detach().cpu()
            logits_fake = -self.discr(Gz).mean()
            (logits_fake - torch.clip(mse_dist, 0, 1)).backward()
            # torch.nn.utils.clip_grad_norm_(self.gen.parameters(), 0.5)
            self.gen_opt.step()
            gen_loss_epoch += logits_fake.item()


        return gen_loss_epoch / n_updates, discr_loss_epoch / n_updates / 5, gen_dist / n_updates

    def generate(self, states):
        with torch.no_grad():
            return self.gen.mu(states)



if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    emb_size = 256
    n_epoch = 1000
    test = False

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(env_id),
        num_processes=32,
        device=device
    )


    vae = VAE(env.observation_space.shape, emb_size=emb_size).to(device)
    gan = GAN(emb_size, device=device)

    if os.path.isfile('vae.pt'):
        vae.load_state_dict(torch.load('vae.pt', map_location=device))
    else:
        train_vae(env, vae)

    if test:
        gan.gen.load_state_dict(torch.load('master.pt', map_location=device))
        gan.discr.load_state_dict(torch.load('discr.pt', map_location=device))

    for epoch in trange(n_epoch):
        with torch.no_grad():
            states = rollout(env)
            states_enc = vae.encode(states)

            if test:
                gen_enc = gan.generate(states_enc)
                dec = vae.decode(gen_enc)
                from matplotlib import pyplot as plt
                for i, d in enumerate(dec):
                    f, axarr = plt.subplots(1, 2)
                    axarr[0].imshow(states[i].permute(1, 2, 0) / 255.)
                    axarr[1].imshow(d.permute(1, 2, 0))
                    plt.show()
                exit(0)


        print(gan.update(states_enc))

        torch.save(gan.gen.state_dict(), 'master.pt')
        torch.save(gan.discr.state_dict(), 'discr.pt')

