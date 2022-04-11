import gym
import torch
from torch import nn
from torch.nn import functional as F
from tqdm import trange
import time
from collections import deque
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from utils import init_params
from rllr.models.ppo import FixedCategorical
from rllr.algo import PPO
from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper
from rllr.utils import get_conf
from rllr.buffer.rollout import RolloutStorage
from rllr.buffer.replay import ReplayBuffer


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


class Wrapper(gym.Wrapper):
    def __init__(self, env):
        super(Wrapper, self).__init__(env)
        self.observation_space = gym.spaces.Box(
            0,
            255,
            (*env.observation_space['image'].shape[:2], env.observation_space['image'].shape[2] * 2)
        )
        self.goal = None

    def observation(self, obs):
        return np.concatenate([obs['image'], self.goal], axis=-1).astype(np.float32) / 255.

    def reset(self):
        self.goal = self.sample_goal()
        return self.observation(self.env.reset())

    def step(self, action):
        obs, _, done, info = self.env.step(action)
        rew = self.calc_reward(obs['image'])
        done = rew > 0
        return self.observation(obs), rew, done, info

    def calc_reward(self, obs):
        img1 = self.goal
        img2 = obs
        #print(img1.shape, img2.shape)
        #from matplotlib import pyplot as plt
        #for diff, i1, i2 in zip(torch.abs(img1.view(img1.size(0), -1) - img2.view(img1.size(0), -1)).sum(dim=1), img1, img2):
        #    fix, ax = plt.subplots(1, 2)
        #    ax[0].imshow(i1)
        #    ax[1].imshow(i2)
        #    plt.title(f'{diff}')
        #    plt.show()
        #img1 = img1.view(img1.size(0), -1)
        #img2 = img2.view(img2.size(0), -1)
        #print(torch.abs(img1 - img2))
        diff = np.abs(img1.reshape(-1) - img2.reshape(-1)).sum(axis=0)
        return float(diff < 0.1)

    def sample_goal(self):
        # probs = agent.get_value(torch.cat([x0, gg], dim=-1)).view(-1)
        #(goal_ids,) = torch.where((probs > 0.1))
        #if len(goal_ids) == 0:
        return GOALS[np.random.randint(0, GOALS.shape[0], (1,))][0]
        # return gg[goal_ids[torch.randint(0, goal_ids.size(0), (16,))]]



def gen_env_with_seed(seed, with_goal=True):
    env = gym.make('MiniGrid-Empty-8x8-v0')
    env.seed(seed)
    env = minigrid_envs.RGBImgObsWrapper(env)
    env = minigrid_envs.FixResetSeedWrapper(env, 0)
    if with_goal:
        env = Wrapper(env)
    return EpisodeInfoWrapper(env)


class PolicyModel(nn.Module):

    def __init__(self, state_shape, n_actions):
        super(PolicyModel, self).__init__()
        self.state_shape = state_shape
        self.n_actions = n_actions
        self.is_recurrent = False

        w, h, c = state_shape
        c = c // 2

        self.enc = nn.Sequential(
            nn.Conv2d(in_channels=c, out_channels=16, kernel_size=4, stride=4),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
        )

        emb_size = self.enc(torch.zeros(1, c, w, h)).size(1)

        self.policy = nn.Sequential(
            nn.Linear(emb_size * 2, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(in_features=512, out_features=self.n_actions)
        )

        self.value = nn.Sequential(
            nn.Linear(emb_size * 2, 512),
            nn.LeakyReLU(inplace=True),
            nn.Linear(512, 1),
            # nn.Sigmoid()
        )

        init_params(self)

    def forward(self, inputs, rnn_hxs, masks):
        inputs = inputs.permute(0, 3, 1, 2)
        state, goal = torch.chunk(inputs, 2, dim=1)
        state_enc = self.enc(state)
        goal_enc = self.enc(goal)
        enc = torch.cat([state_enc, goal_enc], dim=1)
        value = self.value(enc)
        logits = self.policy(enc)
        dist = FixedCategorical(logits=F.log_softmax(logits, dim=1))

        return dist, value, rnn_hxs

    def act(self, states, rnn_hxs=None, masks=None, deterministic=False):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()
        return value, action, dist.log_probs(action), rnn_hxs

    def get_value(self, states, rnn_hxs=None, masks=None):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return value

    def evaluate_actions(self, states, actions, rnn_hxs, masks):
        dist, value, rnn_hxs = self.forward(states, rnn_hxs, masks)
        return value, dist.log_probs(actions), dist.entropy().mean(), rnn_hxs


def init_goal_buffer(conf):
    env = gen_env_with_seed(123, with_goal=False)
    states = [env.reset()['image']]

    for i in trange(conf['training.n_steps'] * 10):
        obs, _, _, _ = env.step(env.action_space.sample())
        states.append(obs['image'])
    return np.asarray(states)


if __name__ == '__main__':
    conf = get_conf()
    device = conf['agent.device']

    GOALS = init_goal_buffer(conf)

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(env_id),
        16,
        device
    )

    agent = PPO(
        actor_critic=PolicyModel(env.observation_space.shape, env.action_space.n),
        clip_param=0.2,
        ppo_epoch=4,
        num_mini_batch=4,
        value_loss_coef=0.5,
        entropy_coef=0.01,
        lr=1e-4,
        eps=1e-5,
        max_grad_norm=0.5
    ).to(device)

    writer = SummaryWriter(conf['outputs.logs'])

    rollouts = RolloutStorage(
        conf['training.n_steps'],
        conf['training.n_processes'],
        env.observation_space,
        env.action_space
    )

    obs = env.reset()
    rollouts.set_first_obs(obs)
    rollouts.to(conf['agent.device'])

    start = time.time()
    num_updates = int(conf['training.n_env_steps'] // conf['training.n_steps'] // conf['training.n_processes'])

    episode_rewards = deque(maxlen=10)

    for j in trange(num_updates):
        update_linear_schedule(agent.optimizer, j, num_updates, conf['agent.lr'])

        for step in range(conf['training.n_steps']):
            # Sample actions
            value, action, action_log_prob, _ = agent.act(obs)
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # idx = (j * conf['training.n_steps'] + step) % goals.size(0)
            # goals[idx] = curr_obs


            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.] if done_ or rew else [1.] for rew, done_ in zip(reward, done)]).to(device)
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        next_value = agent.get_value(rollouts.get_last_obs())
        rollouts.compute_returns(next_value, conf['agent.gamma'], conf['agent.gae_lambda'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        if j % conf['training.verbose'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * conf['training.n_processes'] * conf['training.n_steps']
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

            torch.save(agent, conf['outputs.model'])
