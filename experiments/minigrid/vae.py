import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset
from rllr.utils.logger import init_logger
from torch.utils.tensorboard import SummaryWriter
from rllr.utils import get_conf, switch_reproducibility_on
import rllr.env as environments
from rllr.env import make_vec_envs, EpisodeInfoWrapper
from tqdm import trange, tqdm


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


class VAE(nn.Module):
    def __init__(self, state_shape, emb_size=256):
        super(VAE, self).__init__()
        self.state_shape = state_shape
        self.beta = 5.0

        w, h, c = state_shape

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
            hid = self.enc(x.float().permute(0, 3, 1, 2) / 255.)
            return self.mu(hid)

    def encode_and_sample(self, x):
        with torch.no_grad():
            hid = self.enc(x.float().permute(0, 3, 1, 2) / 255.)
            mu, logvar = self.mu(hid), self.std(hid)
            x = self.sample(mu, logvar)
        return x

    def decode(self, z):
        with torch.no_grad():
            return self.dec(z)

    def sample(self, mu, logvar):
        std = torch.exp(0.5 * logvar)  # e^(1/2 * log(std^2))
        eps = torch.randn_like(std)  # random ~ N(0, 1)
        return eps.mul(std).add_(mu)

    def forward(self, x):
        hid = self.enc(x.float().permute(0, 3, 1, 2) / 255.)
        mu, logvar = self.mu(hid), self.std(hid)
        z = self.sample(mu, logvar)
        rx = self.dec(z)
        return rx, mu, logvar

    def loss(self, recon_x, x, mu, logvar):
        # reconstruction losses are summed over all elements and batch
        recon_loss = F.binary_cross_entropy(recon_x, x.float().permute(0, 3, 1, 2) / 255., reduction='sum')

        # see Appendix B from VAE paper:
        # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
        # https://arxiv.org/abs/1312.6114
        # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        kl_diverge = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        return (recon_loss + self.beta * kl_diverge) / x.shape[0]  # divide total loss by batch size


def gen_env(conf, seed):
    conf['env']['deterministic'] = True
    conf['env']['seed'] = seed
    verbose = 0
    env = environments.minigrid_envs.gen_wrapped_env(conf['env'], verbose=verbose)
    return EpisodeInfoWrapper(env)


def get_data(env, n=1000):
    dataset = list()
    for _ in trange(n):
        obs = env.reset()
        dataset.append(obs)
    dataset = TensorDataset(torch.cat(dataset, dim=0))
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, drop_last=True)
    return dataloader


def train_vae(vae, data, writer, conf):
    optimizer = torch.optim.Adam(vae.parameters())
    total_iter = 0

    for epoch in range(5000):
        train_data = iter(data)

        print("training epoch %d" % epoch)
        for x, in tqdm(data):
            optimizer.zero_grad()
            rx, mu, logvar = vae(x)
            loss = vae.loss(rx, x, mu, logvar)
            loss.backward()
            optimizer.step()
            total_iter += 1
            writer.add_scalar('loss', loss, total_iter)

            if total_iter % 50 == 0:
                writer.add_image('img', x.float().permute(0, 3, 1, 2)[0] / 255., total_iter)
                writer.add_image('rec_img', rx[0], total_iter)
                torch.save(vae, conf['outputs.vae'])


def main(args=None):
    conf = get_conf(args)
    switch_reproducibility_on(conf['seed'])
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    env = make_vec_envs(lambda env_id: lambda: gen_env(conf, env_id),
                        10,
                        conf['training.device'])
    data = get_data(env)

    vae = VAE(env.observation_space.shape, 256).to(device)

    writer = SummaryWriter(conf['outputs.vae_logs'])
    train_vae(vae, data, writer, conf)


if __name__ == "__main__":
    torch.autograd.set_detect_anomaly(True)
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
