import logging

import numpy as np
import torch

from argparse import ArgumentParser
from pyhocon.config_parser import ConfigFactory
from tqdm import trange


from rllr.env.vec_wrappers import make_vec_envs
from rllr.utils.logger import init_logger
import torch.nn as nn
from einops import rearrange
from train_rnd_ppo import PolicyModel, TargetModel, PredictorModel, Encoder, RNNEncoder


logger = logging.getLogger(__name__)


def main(args):
    if args.mode == 'rnd_ppo':
        from train_rnd_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('../montezuma/conf/montezuma_rnd_ppo.hocon')

    agent = torch.load(config['outputs.path'], map_location='cpu')

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, np.random.randint(0, 10000000), render=args.viz),
        num_processes=1,
        device='cpu'
    )

    rewards, steps, successes = [], [], []
    for _ in trange(args.episodes):
        obs, done, episode_reward = env.reset(), False, 0
        episode_steps = 0
        rnn_hxs = torch.zeros((1, config.get('encoder.recurrent_hidden_size', 1)))
        masks = torch.ones((1, 1))

        while not done:
            value, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=False)
            # observation, reward and next obs
            obs, reward, done, _ = env.step(action)
            episode_reward += float(reward)
            episode_steps += 1
        rewards.append(episode_reward)
        steps.append(episode_steps)
        successes.append(episode_reward > 0)
        if args.viz:
            input(f'> the end!, reward = {episode_reward}')

    logger.info(f'mode {args.mode}: '
                f'reward {np.mean(rewards):.2f}, '
                f'steps {np.mean(steps):.2f}, '
                f'success {np.mean(successes):.2f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['worker', 'master', 'ssim_master', 'ssim_worker', 'direct_ppo', 'rnd_ppo', 'multi', 'monte'])
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--episodes', default=100, type=int)
    args = parser.parse_args()

    init_logger(__name__)
    main(args)
