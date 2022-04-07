import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
import numpy as np
import gym
from gym_minigrid.wrappers import RGBImgObsWrapper
from collections import deque
from tqdm import trange
import time
from tensorboardX import SummaryWriter
from sklearn.neighbors import KDTree
from matplotlib import pyplot as plt

from rllr.algo import PPO
from rllr.models.ppo import FixedCategorical
from rllr.env import make_vec_envs, EpisodeInfoWrapper
from rllr.utils import train_ppo, get_conf
from rllr.env.gym_minigrid_navigation.environments import ImageObsWrapper
from rllr.buffer.rollout import RolloutStorage


GOAL_SIZE = 256


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


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


class GoalWrapper(gym.Wrapper):
    def __init__(self, env):
        super(GoalWrapper, self).__init__(env)
        img_shape = env.observation_space['image'].shape
        self.observation_space = gym.spaces.Box(0, 255, (img_shape[2], img_shape[0], img_shape[1]))
        self.action_space = gym.spaces.Discrete(3)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.observation(obs), reward, done, info

    def reset(self, **kwargs):
        self.env.seed(0)
        return self.observation(super().reset())

    def observation(self, obs):
        return obs['image'].transpose(2, 0, 1)


def gen_env_with_seed(conf, seed):
    env = gym.make('MiniGrid-LavaCrossingS9N1-v0')
    env.seed(seed=seed)
    env = RGBImgObsWrapper(env, tile_size=8)
    env = GoalWrapper(env)
    env = EpisodeInfoWrapper(env)
    return env


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super(Encoder, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=4, stride=4),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
        )

        out_shape = self.conv(torch.zeros(1, obs_shape[2], obs_shape[0], obs_shape[1])).shape
        conv_out = out_shape[1] * out_shape[2] * out_shape[3]

        self.fc = nn.Sequential(
            nn.Linear(in_features=conv_out, out_features=1024),
            nn.ReLU(inplace=True),
        )

        self.output_size = 128

    def forward(self, t):
        hid = t.permute(0, 3, 1, 2).float() / 255.
        hid = self.conv(hid).reshape(t.shape[0], -1)
        hid = self.fc(hid)
        return hid


class PolicyModel(nn.Module):

    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.is_recurrent = False

        self.enc = Encoder(self.state_shape)

        self.policy = nn.Sequential(
            nn.Linear(in_features=self.enc.output_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=self.n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(in_features=self.enc.output_size, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=1)
        )

        self.apply(init_params)

    def forward(self, inputs, rnn_hxs, masks):
        enc = self.enc(inputs)
        value = self.value(enc)
        logits = self.policy(enc)
        dist = FixedCategorical(logits=F.log_softmax(logits, dim=1))

        return dist, value, rnn_hxs

    def act(self, states, rnn_hxs, masks, deterministic=False):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return value, action, dist.log_probs(action), rnn_hxs

    def get_value(self, states, rnn_hxs, masks):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return value

    def evaluate_actions(self, states, actions, rnn_hxs, masks):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return value, dist.log_probs(actions), dist.entropy().mean(), rnn_hxs


class VAEModel(nn.Module):
    def __init__(self, state_shape):
        super(VAEModel, self).__init__()
        self.state_shape = state_shape
        self.output_size = 256
        self.beta = 1.0

        c, w, h = state_shape
        print(c, w, h)

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(in_features=5184, out_features=256),
        )

        self.mu = nn.Linear(in_features=256, out_features=256)
        self.std = nn.Linear(in_features=256, out_features=256)

        self.dec = nn.Sequential(
            nn.Linear(256, 5184),
            nn.Unflatten(dim=1, unflattened_size=(16, 18, 18)),
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, padding=1),
            nn.LeakyReLU(inplace=True),

            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(in_channels=16, out_channels=3, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        self.apply(init_params)

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


class VAE:
    def __init__(self, state_shape, batch_size=256):
        self.vae_model = VAEModel(state_shape)
        self.batch_size = batch_size
        self.opt = optim.Adam(self.vae_model.parameters())

    def update(self, observations):
        observations = observations / 255.
        epochs = observations.size(0) // self.batch_size
        mean_loss = 0
        for i in range(epochs):
            ids = torch.randint(0, observations.size(0), (self.batch_size,))
            imgs = observations[ids]
            recon_imgs, mu, logvar = self.vae_model(imgs)
            self.opt.zero_grad()
            loss = self.vae_model.loss(recon_imgs, imgs, mu, logvar)
            loss.backward()
            self.opt.step()
            mean_loss += loss.detach().cpu().numpy()
        return mean_loss / epochs

    def to(self, device):
        self.vae_model = self.vae_model.to(device)
        return self


def pretrain(env, config):
    obs = env.reset()
    vae = VAE(env.observation_space.shape)
    observations = torch.zeros((config['training.n_steps'] + 1, *obs.shape))
    for j in trange(500):
        for step in range(config['training.n_steps']):
            # Sample actions
            obs, reward, done, infos = env.step(torch.randint(0, env.action_space.n, (obs.size(0), 1)))
            observations[step] = obs

        loss = vae.update(observations[:-1].view(-1, *obs.shape[1:]))
        print(loss)
        observations[0] = observations[-1]
        torch.save(vae, 'vae.p')
    return vae


def test(env, config, ssim):
    def encode(t):
        with torch.no_grad():
            return ssim.ssim_model.enc(t).numpy()

    def sim(emb1, emb2):
        emb1 = torch.from_numpy(emb1).float()
        emb2 = torch.from_numpy(emb2).float()
        with torch.no_grad():
            return ssim.ssim_model.fc(torch.cat([emb1, emb2], dim=1)).numpy()

    obs = env.reset()
    observations = []
    images = []
    observations.extend(encode(obs))
    images.extend(obs)
    for j in trange(10):
        for step in range(config['training.n_steps']):
            # Sample actions
            obs, reward, done, infos = env.step(torch.randint(0, env.action_space.n, (obs.size(0), 1)))
            observations.extend(encode(obs))
            images.extend(obs.numpy())
    observations = np.asarray(observations)
    images = np.asarray(images)
    print('obs.shape', observations.shape)
    print('img.shape', images.shape)
    print('hui', images[-1].shape)
    tree = KDTree(observations)
    for i in range(20):
        enc = torch.randn((1, 1024))
        enc = (enc - enc.mean(axis=1).unsqueeze(-1)) / (enc.pow(2).sum(dim=1)).pow(0.5).unsqueeze(-1).expand(*enc.size())
        enc = enc.numpy()

        print((enc ** 2).sum(axis=1))
        dist, ind = tree.query(enc, k=3)
        dist, ind = dist[0], ind[0]
        fig, ax = plt.subplots(1, 3)
        ax[0].imshow(images[ind[0]])
        ax[0].set_title(f'L2 = {dist[0]}, sim = {sim(enc, observations[ind[0]].reshape(1, 1024))}')
        ax[1].imshow(images[ind[1]])
        ax[1].set_title(f'L2 = {dist[1]}, sim = {sim(enc, observations[ind[1]].reshape(1, 1024))}')
        ax[2].imshow(images[ind[2]])
        ax[2].set_title(f'L2 = {dist[2]}, sim = {sim(enc, observations[ind[2]].reshape(1, 1024))}')
        plt.show()


if __name__ == '__main__':
    config = get_conf()

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    worker_policy = PolicyModel(env.observation_space.shape, env.action_space.n)
    worker_policy.to(config['agent.device'])

    worker = PPO(
        worker_policy,
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm']
    )

    master_policy = PolicyModel(env.observation_space.shape, env.action_space.n)
    master_policy.to(config['agent.device'])

    master = PPO(
        worker_policy,
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm']
    )

    writer = SummaryWriter(config['outputs.logs'])

    rollouts = RolloutStorage(
        config['training.n_steps'], config['training.n_processes'], env.observation_space, env.action_space
    )

    obs = env.reset()
    rollouts.set_first_obs(obs)
    rollouts.to(config['agent.device'])

    start = time.time()
    num_updates = int(config['training.n_env_steps'] // config['training.n_steps'] // config['training.n_processes'])

    episode_rewards = deque(maxlen=10)

    for j in trange(num_updates):
        update_linear_schedule(worker.optimizer, j, num_updates, config['agent.lr'])

        for step in range(config['training.n_steps']):
            # Sample actions
            value, action, action_log_prob, _ = worker.act(obs)
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        next_value = worker.get_value(rollouts.get_last_obs())
        rollouts.compute_returns(next_value, config['agent.gamma'], config['agent.gae_lambda'])

        value_loss, action_loss, dist_entropy = worker.update(rollouts)
        rollouts.after_update()

        if j % config['training.verbose'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config['training.n_processes'] * config['training.n_steps']
            end = time.time()
            print(f'Updates {j}, '
                  f'num timesteps {total_num_steps}, '
                  f'FPS {int(total_num_steps / (end - start))} \n'
                  f'Last {len(episode_rewards)} training episodes: '
                  f'mean/median reward {np.mean(episode_rewards):.2f}/{np.median(episode_rewards):.2f}, '
                  f'min/max reward {np.min(episode_rewards):.2f}/{np.max(episode_rewards):.2f}\n'
                  f'dist_entropy {dist_entropy:.2f}, '
                  f'value_loss {value_loss:.2f}, '
                  f'action_loss {action_loss:.2f}')

            writer.add_scalar('dist_entropy', dist_entropy, total_num_steps)
            writer.add_scalar('value_loss', value_loss, total_num_steps)
            writer.add_scalar('action_loss', action_loss, total_num_steps)
            writer.add_scalar('reward', np.mean(episode_rewards), total_num_steps)

            torch.save(worker, config['outputs.model'])
