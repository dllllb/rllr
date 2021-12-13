import logging

import numpy as np
import torch

from argparse import ArgumentParser
from pyhocon.config_parser import ConfigFactory
from tqdm import trange

from rllr.env import make_vec_envs
from rllr.utils.logger import init_logger

logger = logging.getLogger(__name__)


def main(args):
    if args.mode == 'worker':
        from experiments.minigrid.train_worker import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_first_step.hocon')
        agent_path = 'artifacts/models/minigrid_worker.p'

    elif args.mode == 'master':
        from experiments.minigrid.train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_second_step.hocon')
        agent_path = 'artifacts/models/minigrid_master.p'

    elif args.mode == 'ssim_worker':
        from experiments.minigrid.train_worker import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_first_step_ssim.hocon')
        agent_path = 'artifacts/models/minigrid_worker_ssim.p'

    elif args.mode == 'ssim_master':
        from experiments.minigrid.train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_second_step_ssim.hocon')
        agent_path = 'artifacts/models/minigrid_master_ssim.p'

    elif args.mode == 'direct_ppo':
        from experiments.minigrid.train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_direct_ppo.hocon')
        config['env']['agent_start_pos'] = (1, 1)
        agent_path = 'artifacts/models/minigrid_direct_ppo.p'

    elif args.mode == 'rnd_ppo':
        from train_rnd_gru_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_rnd_ppo.hocon')
        agent_path = 'artifacts/models/minigrid_rnd_ppo.p'

    elif args.mode == 'multi':
        from train_multitask import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_multitask.hocon')

    agent = torch.load(config['outputs.path'], map_location='cpu')

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, 11),
        num_processes=1,
        device='cpu'
    )

    rewards, steps, successes = [], [], []
    for _ in trange(args.episodes):
        obs, done, episode_reward = env.reset(), False, 0
        episode_steps = 0
        rnn_hxs = torch.zeros((1, 256))
        masks = torch.zeros((1, 256))

        while not done:
            if args.viz:
                env.render('human')
            print(rnn_hxs, obs.shape)
            value, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=True)
            print(action, env.action_space.n)
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
    parser.add_argument('--mode', choices=['worker', 'master', 'ssim_master', 'ssim_worker', 'direct_ppo', 'rnd_ppo', 'multi'])
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--episodes', default=100, type=int)
    args = parser.parse_args()

    init_logger(__name__)
    main(args)
