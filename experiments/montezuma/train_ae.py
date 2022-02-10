from pyhocon import ConfigFactory
from train_rnd_ppo import gen_env_with_seed
from rllr.env.vec_wrappers import make_vec_envs
import torch
import numpy as np
import torch.nn as nn
from torch.optim import AdamW
from matplotlib import pyplot as plt
from tqdm import trange, tqdm


def rollout(env, agent, config):
    obs, done, info = env.reset(), False, []
    rnn_hxs = torch.zeros((1, config.get('encoder.recurrent_hidden_size', 1))).to(device)
    masks = torch.ones((1, 1)).to(device)

    states = []
    while not done:
        states.append(obs)
        value, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=False)
        obs, reward, done, info = env.step(action)
    states.append(obs)

    return torch.cat(states, dim=0)


class AE(nn.Module):
    def __init__(self, state_shape):
        super(AE, self).__init__()
        self.state_shape = state_shape
        self.output_size = 256

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

    def forward(self, t):
        return self.dec(self.enc(t))


def train_ae(env, agent, config):
    n_epoch = 1000
    batch_size = 32
    loss_fn = nn.MSELoss()
    opt = AdamW(aenc.parameters())

    for epoch in trange(n_epoch):
        rec_loss = []
        states = rollout(env, agent, config)
        for _ in range(states.shape[0] // batch_size):
            ids = torch.randint(0, states.shape[0], (batch_size,))
            imgs = (states[ids] / 255.).to(device)
            rec = aenc(imgs)

            opt.zero_grad()
            loss = loss_fn(rec, imgs)
            loss.backward()
            opt.step()

            rec_loss.append(loss.detach().cpu().numpy())
        print(np.mean(rec_loss))
        torch.save(aenc.state_dict(), 'aenc.p')


def test_ae(env, agent, config):
    states = rollout(env, agent, config)
    ids = torch.randint(0, states.shape[0] - 1, (states.shape[0],))
    for i in tqdm(ids):
        img = states[i].unsqueeze(dim=0) / 255.

        with torch.no_grad():
            rec = aenc(img)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img[0].permute(1, 2, 0), cmap='gray')
        axarr[1].imshow(rec[0].permute(1, 2, 0), cmap='gray')
        plt.show()


if __name__ == '__main__':
    device = 'cuda:0'
    config = ConfigFactory.parse_file('conf/montezuma_rnd_ppo.hocon')

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, 0, render=False),
        num_processes=1,
        device=device
    )

    agent = torch.load(config['outputs.path'], map_location=device)
    aenc = AE(env.observation_space.shape).to(device)
    train_ae(env, agent, config)

    aenc.load_state_dict(torch.load(open('aenc.p', 'rb'), map_location=device))
    test_ae(env, agent, config)
