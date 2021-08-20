import os
import sys
import numpy as np
import torch
import pickle

from pyhocon.config_parser import ConfigFactory
from rllr.env.vec_wrappers import make_vec_envs
from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['worker', 'master'])
    args = parser.parse_args()

    if args.mode == 'master':
        from train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_second_step.hocon')
    else:
        from train_worker import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_first_step.hocon')

    agent = torch.load(config['outputs.path'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        num_processes=1,
        device='cpu'
    )

    while True:
        obs = env.reset()
        done = False
        cum_reward = 0

        while not done:
            env.render('human')
            value, action, _ = agent.act(obs, deterministic=True)
            # Obser reward and next obs
            obs, reward, done, _ = env.step(action)
            cum_reward += reward

        input(f'> the end!, reward = {cum_reward}')
