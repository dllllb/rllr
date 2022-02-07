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

    states, actions = [], []
    while not done:
        states.append(obs)
        value, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=False)
        obs, reward, done, info = env.step(action)
        actions.append(action)
    states.append(obs)

    return torch.cat(states, dim=0), torch.cat(actions, dim=0).view(-1)


class AE(nn.Module):
    def __init__(self, state_shape, n_actions):
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

        self.inv = nn.Sequential(
            nn.Linear(256 * 2, 128),
            nn.LeakyReLU(inplace=True),
            nn.Linear(128, n_actions),
            nn.LogSoftmax(dim=1)
        )

        for layer in self.modules():
            if isinstance(layer, nn.Conv2d):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                layer.bias.data.zero_()

    def forward(self, t1, t2):
        hid1 = self.enc(t1)
        hid2 = self.enc(t2)
        return self.dec(hid1), self.inv(torch.cat([hid1, hid2], dim=1))


if __name__ == '__main__':
    device = 'cpu'
    config = ConfigFactory.parse_file('conf/montezuma_rnd_ppo.hocon')

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, 0, render=False),
        num_processes=1,
        device=device
    )

    agent = torch.load(config['outputs.path'], map_location=device)
    aenc = AE(env.observation_space.shape, env.action_space.n).to(device)
    n_epoch = 1000
    batch_size = 32
    mse_loss_fn = nn.MSELoss()
    nll_loss_fn = nn.NLLLoss()
    opt = AdamW(aenc.parameters())
    losses = []

    for epoch in trange(n_epoch):
        rec_loss = []
        inv_loss = []
        states, actions = rollout(env, agent, config)
        for _ in range(states.shape[0] // batch_size):
            ids = torch.randint(0, states.shape[0] - 1, (batch_size,))
            imgs1 = (states[ids] / 255.).to(device)
            imgs2 = (states[ids + 1] / 255.).to(device)

            rec, logits = aenc(imgs1, imgs2)

            opt.zero_grad()
            mse_loss = mse_loss_fn(rec, imgs1)
            nll_loss = nll_loss_fn(logits, actions[ids])
            loss = mse_loss + nll_loss * 0.01
            loss.backward()
            opt.step()

            rec_loss.append(mse_loss.detach().cpu().numpy())
            inv_loss.append(nll_loss.detach().cpu().numpy())
        print(np.mean(rec_loss), np.mean(inv_loss))
        torch.save(aenc.state_dict(), 'aenc.p')

    aenc.load_state_dict(torch.load(open('aenc.p', 'rb'), map_location=device))
    states, actions = rollout(env, agent, config)
    ids = torch.randint(0, states.shape[0] - 1, (states.shape[0],))
    for i in tqdm(ids):
        img1 = states[i].unsqueeze(dim=0) / 255.
        img2 = states[i + 1].unsqueeze(dim=0) / 255.

        with torch.no_grad():
            rec, logits = aenc(img1, img2)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img1[0].permute(1, 2, 0), cmap='gray')
        axarr[1].imshow(rec[0].permute(1, 2, 0), cmap='gray')
        plt.show()
