import logging

import numpy as np
import torch
from argparse import ArgumentParser
from pyhocon.config_parser import ConfigFactory
from tqdm import trange

from rllr.env import make_vec_envs
from rllr.utils.logger import init_logger
from rllr.utils import switch_reproducibility_on

logger = logging.getLogger(__name__)


def test():
    algos = {
        'worker': (2.0, 1.0),
        'master': (0.90156251, 1.0),
        'ssim_master': (0.936718762, 1.0),
        'ssim_master_lava': (0.95499998, 1.0),
        'ssim_worker': (0, 0),
        'direct_ppo': (0.87695312, 1.0),
        'direct_ppo_lava': (0.94722223, 1.0),
        'rnd_ppo': (0.30742186, 1.0),
        'rnd_ppo_lava': (0.86111110, 1.0),
        'rnd_ppo_doorkey': (0.960625, 1.0),
        'rnd_ppo_fourrooms': (0.73, 1.0),
        'multi': (0.77222222, 1.0)
    }

    for algo, (expected_reward, expected_success) in algos.items():
        switch_reproducibility_on()
        rewards, steps, success = play(mode=algo, viz=False, n_episodes=1)
        assert np.allclose(rewards, expected_reward)
        assert np.allclose(success, expected_success)


def play(mode, viz, n_episodes):
    if mode == 'worker':
        from train_worker import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_first_step.hocon')

    elif mode == 'master':
        from train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_second_step.hocon')

    elif mode == 'ssim_worker':
        from train_worker import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_first_step_ssim.hocon')

    elif mode == 'ssim_master':
        from train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_second_step_ssim.hocon')

    elif mode == 'ssim_master_lava':
        from train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_lava_second_step_ssim.hocon')

    elif mode == 'direct_ppo':
        from train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_direct_ppo.hocon')
        config['env']['agent_start_pos'] = (1, 1)

    elif mode == 'direct_ppo_lava':
        from train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_direct_ppo_lava.hocon')

    elif mode == 'rnd_ppo':
        from train_rnd_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_rnd_ppo.hocon')

    elif mode == 'rnd_ppo_lava':
        from train_rnd_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_rnd_ppo_lava.hocon')

    elif mode == 'rnd_ppo_doorkey':
        from train_rnd_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_rnd_ppo_doorkey.hocon')

    elif mode == 'rnd_ppo_fourrooms':
        from train_rnd_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_rnd_ppo_fourrooms.hocon')

    elif mode == 'multi':
        from train_multitask import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_multitask.hocon')

    else:
        assert False

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        num_processes=1,
        device='cpu'
    )

    agent = torch.load(config['outputs.model'], map_location='cpu')

    rewards, steps, successes = [], [], []
    for _ in trange(n_episodes):
        obs, done, episode_reward = env.reset(), False, 0
        rnn_hxs = torch.zeros((1, config.get('encoder.recurrent_hidden_size', 1)))
        masks = torch.ones((1, 1))

        while not done:
            if viz:
                env.render('human')
            value, action, _, rnn_hxs = agent.act(obs, rnn_hxs, masks, deterministic=False)
            obs, reward, done, infos = env.step(action)
            episode_reward += float(reward)

        rewards.append(episode_reward)
        steps.append(infos[0]['episode']['steps'])
        successes.append(episode_reward > 0)
        if viz:
            input(f'> the end!, reward = {episode_reward}')
    return np.mean(rewards), np.mean(steps), np.mean(successes)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', choices=['worker', 'master', 'ssim_master', 'ssim_master_lava',
                                           'ssim_worker', 'direct_ppo', 'rnd_ppo', 'rnd_ppo_doorkey',
                                           'rnd_ppo_fourrooms', 'multi'])
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--episodes', default=100, type=int)
    args = parser.parse_args()

    init_logger(__name__)
    rewards, steps, successes = play(args.mode, args.viz, args.episodes)
    logger.info(f'mode {args.mode}: '
                f'reward {rewards:.2f}, '
                f'steps {steps:.2f}, '
                f'success {successes:.2f}')
