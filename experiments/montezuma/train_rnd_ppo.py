import logging

import gym
from rllr.env.gym_minigrid_navigation import environments as minigrid_envs
from rllr.utils import im_train_ppo
from rllr.env.vec_wrappers import make_vec_envs
from rllr.utils import switch_reproducibility_on, get_conf
from rllr.env.wrappers import EpisodeInfoWrapper
from rllr.models import encoders
from rllr.models.ppo import ActorCriticNetwork
from rllr.algo import IMPPO
from rllr.models import RNDModel
from rllr.utils.logger import init_logger
import torch.nn as nn
from rllr.models.encoders import RNNEncoder
from einops import rearrange
import cv2
import numpy as np


logger = logging.getLogger(__name__)


class MontezumaInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(MontezumaInfoWrapper, self).__init__(env)
        self.room_address = 3
        self.visited_rooms = set()
        self.observation_space = gym.spaces.Box(0, 255, (84, 84, 3), dtype=np.uint8)

        self.episode_reward = 0
        self.episode_steps = 0

    def get_current_room(self):
        ram = self.unwrapped.ale.getRAM()
        assert len(ram) == 128
        return int(ram[self.room_address])

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.episode_reward += rew
        self.episode_steps += 1
        self.visited_rooms.add(self.get_current_room())

        if done:
            info['episode'] = {
                'r': self.episode_reward,
                'steps': self.episode_steps,
                'task': 'Montezuma',
                'visited_rooms':  len(self.visited_rooms)
            }
            self.visited_rooms.clear()

        return self.observation(obs), rew, done, info

    def reset(self):
        self.episode_reward = 0
        self.episode_steps = 0
        return self.observation(self.env.reset())

    def observation(self, t):
        return cv2.resize(t, (84, 84))


def gen_env_with_seed(conf, seed):
    env = gym.make(conf['env.env_id'])
    env.seed(seed)
    return MontezumaInfoWrapper(env)


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.feature = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=4, stride=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=4, stride=4, padding=0, dilation=1, ceil_mode=False),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        self.output_size = 400

    def forward(self, t):
        t = rearrange(t, 'b h w c -> b c h w')
        x =  self.feature(t)
        return x.view(t.shape[0], -1)


def get_agent(env, config):
    state_conf = config['encoder']
    hidden_size = 400
    state_encoder = RNNEncoder(Encoder(), recurrent_hidden_size=hidden_size)
    policy = ActorCriticNetwork(
        env.action_space, state_encoder,
        state_encoder, hidden_size,
        hidden_size, use_intrinsic_motivation=True
    )

    rnd_enc = Encoder()
    rnd_pred = Encoder()

    rnd = RNDModel(
        rnd_enc,
        rnd_pred,
        config['agent.device'])

    return IMPPO(
        policy,
        rnd,
        config['agent.ext_coef'],
        config['agent.im_coef'],
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm']
    )


def main(args=None):
    config = get_conf(args)

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    agent = get_agent(env, config)
    agent.to(config['agent.device'])

    logger.info(f"Running agent training: { config['training.n_steps'] * config['training.n_processes']} steps")
    im_train_ppo(
        env=env,
        agent=agent,
        conf=config,
    )


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
