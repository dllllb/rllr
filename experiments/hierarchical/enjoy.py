import logging

import numpy as np
import torch

from argparse import ArgumentParser
from tqdm import trange

from rllr.env.vec_wrappers import make_vec_envs
from rllr.utils.logger import init_logger
from matplotlib import pyplot as plt

logger = logging.getLogger(__name__)


def play(mode, viz, n_episodes):
    from train import gen_env_with_seed

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(seed=np.random.randint(0, 100)),
        num_processes=1,
        device='cpu'
    )

    agent = torch.load('artifacts/models/master.p', map_location='cpu')

    rewards, steps, rooms = [], [], []
    for _ in trange(n_episodes):
        obs, done = env.reset(), False
        masks = torch.ones((1, 1))

        while not done:
            value, action, _, rnn_hxs = agent.act(obs, masks, deterministic=False)
            # observation, reward and next obs
            obs, reward, done, info = env.step(action)
            if args.viz:
                env.render()
        episode = info[0]['episode']
        rewards.append(episode['r'])
        steps.append(episode['steps'])
        if viz:
            input(f'> the end!, reward = {rewards[-1]}')

    return np.mean(rewards), np.mean(steps)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['rnd_ppo'])
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--episodes', default=100, type=int)
    args = parser.parse_args()

    init_logger(__name__)
    rewards, steps = play(args.mode, args.viz, args.episodes)

    logger.info(f'mode {args.mode}: '
                f'reward {np.mean(rewards):.2f}, '
                f'steps {np.mean(steps):.2f}')
