import logging
from pathlib import Path

import numpy as np
import torch

from argparse import ArgumentParser
from pyhocon.config_parser import ConfigFactory
from tqdm import trange

from rllr.env.vec_wrappers import make_vec_envs
from rllr.utils.logger import init_logger
from rllr.utils import switch_reproducibility_on

logger = logging.getLogger(__name__)


def test():
    algos = {
        'rnd_ppo': (6500, 12)
    }

    for algo, (expected_reward, expected_rooms) in algos.items():
        switch_reproducibility_on()
        reward, steps, rooms = play(mode=algo, viz=False, n_episodes=1)
        assert np.allclose(reward, expected_reward)
        assert np.allclose(rooms, expected_rooms)


def play(mode, viz, n_episodes):
    if mode == 'rnd_ppo':
        from train_rnd_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file(Path(__file__).parent / 'conf/montezuma_rnd_ppo.hocon')
    else:
        assert False

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(seed=np.random.randint(0, 100), render=viz),
        num_processes=1,
        device='cpu'
    )

    agent = torch.load(Path(__file__).parent / config['outputs.model'], map_location='cpu')

    rewards, steps, rooms = [], [], []
    for _ in trange(n_episodes):
        obs, done = env.reset(), False
        rnn_hxs = torch.zeros((1, config.get('encoder.recurrent_hidden_size', 1)))
        masks = torch.ones((1, 1))

        while not done:
            value, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=False)
            # observation, reward and next obs
            obs, reward, done, info = env.step(action)
        episode = info[0]['episode']
        rewards.append(episode['r'])
        steps.append(episode['steps'])
        rooms.append(episode['visited_rooms'])
        if viz:
            input(f'> the end!, reward = {rewards[-1]}')

    return np.mean(rewards), np.mean(steps), np.mean(rooms)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['rnd_ppo'])
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--episodes', default=100, type=int)
    args = parser.parse_args()

    init_logger(__name__)
    rewards, steps, rooms = play(args.mode, args.viz, args.episodes)

    logger.info(f'mode {args.mode}: '
                f'reward {np.mean(rewards):.2f}, '
                f'steps {np.mean(steps):.2f}, '
                f'rooms {np.mean(rooms):.2f}')
