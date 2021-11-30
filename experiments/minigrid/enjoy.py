import os
import sys
import numpy as np
import torch
import pickle
from tqdm import trange

from pyhocon.config_parser import ConfigFactory
from rllr.env.vec_wrappers import make_vec_envs
from argparse import ArgumentParser


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['worker', 'master', 'ssim_master', 'ssim_worker'])
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--episodes', default=100, type=int)
    args = parser.parse_args()

    if args.mode == 'master':
        from train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_second_step.hocon')
        agent_path = 'artifacts/models/minigrid_master.p'
    elif args.mode == 'ssim_master':
        from train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_second_step.hocon')
        agent_path = 'artifacts/models/minigrid_master_ssim.p'
    elif args.mode == 'ssim_worker':
        from train_worker import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_first_step.hocon')
        agent_path = 'artifacts/models/minigrid_worker_ssim.p'
    elif args.mode == 'worker':
        from train_worker import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_first_step.hocon')
        agent_path = 'artifacts/models/minigrid_worker.p'

    agent = torch.load(agent_path)

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        num_processes=1,
        device='cpu'
    )

    rewards, steps, successes = [], [], []
    for episode_id in trange(args.episodes):
        obs, done, episode_reward = env.reset(), False, 0
        episode_steps = 0

        while not done:
            if args.viz:
                env.render('human')
            value, action, _ = agent.act(obs, deterministic=True)
            # Obser reward and next obs
            obs, reward, done, _ = env.step(action)
            episode_reward += float(reward)
            episode_steps += 1
        rewards.append(episode_reward)
        steps.append(episode_steps)
        successes.append(episode_reward > 0)
        if args.viz:
            input(f'> the end!, reward = {episode_reward}')

    print(f'mode {args.mode}: '
          f'reward {np.mean(rewards):.2f} +- {np.std(rewards):.2f}, '
          f'steps {np.mean(steps):.2f} +- {np.std(steps):.2f}, '
          f'success {np.mean(successes):.2f} +- {np.std(successes):.2f}'
    )
