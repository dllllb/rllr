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
    rnn_hxs = torch.zeros((1, config.get('encoder.recurrent_hidden_size', 1)))
    masks = torch.ones((1, 1))

    states = []
    while not done:
        states.append(obs)
        value, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=False)
        obs, reward, done, info = env.step(action)
    states.append(obs)

    return torch.from_numpy(np.concatenate(states, axis=0))


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


if __name__ == '__main__':
    config = ConfigFactory.parse_file('conf/montezuma_rnd_ppo.hocon')

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, 0, render=False),
        num_processes=1,
        device='cpu'
    )

    agent = torch.load(config['outputs.path'], map_location='cpu')
    aenc = AE(env.observation_space.shape)
    n_epoch = 0
    batch_size = 32
    loss_f = nn.MSELoss()
    opt = AdamW(aenc.parameters())
    losses = []

    for epoch in trange(n_epoch):
        epoch_loss = []
        dat = rollout(env, agent, config)
        for _ in range(dat.shape[0] // batch_size):
            ids = torch.randint(0, dat.shape[0], (batch_size,))
            imgs = dat[ids] / 255.
            rec = aenc(imgs)

            opt.zero_grad()
            loss = loss_f(rec, imgs)
            loss.backward()
            opt.step()

            epoch_loss.append(loss.detach().cpu().numpy())
        print(np.mean(epoch_loss))
        torch.save(aenc.state_dict(), 'aenc.p')

    aenc.load_state_dict(torch.load(open('aenc.p', 'rb'), map_location='cpu'))
    dat = rollout(env, agent, config)
    ids = torch.randint(0, dat.shape[0], (dat.shape[0],))
    for img in tqdm(dat[ids]):
        img = img.unsqueeze(dim=0) / 255.
        with torch.no_grad():
            rec = aenc(img)
        f, axarr = plt.subplots(1, 2)
        axarr[0].imshow(img[0].permute(1, 2, 0), cmap='gray')
        axarr[1].imshow(rec[0].permute(1, 2, 0), cmap='gray')
        plt.show()
