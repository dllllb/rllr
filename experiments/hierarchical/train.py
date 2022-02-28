import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import gym
from gym_minigrid.wrappers import RGBImgObsWrapper
from collections import deque
from tqdm import trange
import time
from tensorboardX import SummaryWriter

from rllr.algo import PPO
from rllr.models.ppo import FixedCategorical
from rllr.env import make_vec_envs, EpisodeInfoWrapper
from rllr.utils import train_ppo, get_conf
from rllr.env.gym_minigrid_navigation.environments import ImageObsWrapper
from rllr.buffer.rollout import RolloutStorage


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def gen_env_with_seed(conf, seed):
    env = gym.make('MiniGrid-LavaCrossingS9N1-v0')
    env.seed(seed=seed)
    env = RGBImgObsWrapper(env, tile_size=8)
    env = ImageObsWrapper(env)
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

        self.output_size = 1024

    def forward(self, t):
        hid = t.permute(0, 3, 1, 2).float() / 255.
        hid = self.conv(hid).view(t.shape[0], -1)
        return self.fc(hid)


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


if __name__ == '__main__':
    config = get_conf()

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    policy = PolicyModel(env.observation_space.shape, env.action_space.n)
    agent = PPO(
        policy,
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm']
    )

    # train_ppo(env, agent, config)
    # exit(0)

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
        update_linear_schedule(agent.optimizer, j, num_updates, config['agent.lr'])

        for step in range(config['training.n_steps']):
            # Sample actions
            value, action, action_log_prob, _ = agent.act(obs)
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        next_value = agent.get_value(rollouts.get_last_obs())
        rollouts.compute_returns(next_value, config['agent.gamma'], config['agent.gae_lambda'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
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

            torch.save(agent, config['outputs.model'])
