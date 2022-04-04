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
        'ppo_dynobs': (0.0, 0.0),
        'ppo_lava': (0.952777, 1.0),
        'ppo_doorkey': (0.983124, 1.0),
        'ppo_fourrooms': (0.802, 1.0),
        'ppo_keycorridor': (0.92, 1.0),
        'ppo_putnear': (0.79, 1.0),
        'rnd_ppo': (0.760937, 1.0),
        'rnd_ppo_lava': (0.908333, 1.0),
        'rnd_ppo_doorkey': (0.953593, 1.0),
        'rnd_ppo_fourrooms': (0.73, 1.0),
        'rnd_ppo_keycorridor': (0.7233333, 1.0),
        'rnd_ppo_putnear': (0.79, 1.0),
        'multi': (0.77222222, 1.0)
    }

    for algo, (expected_reward, expected_success) in algos.items():
        switch_reproducibility_on()
        rewards, steps, success = play(mode=algo, viz=False, n_episodes=1)
        print(algo, rewards, steps,success)
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

    elif mode == 'worker_doorkey':
        from train_worker import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_first_step_doorkey.hocon')

    elif mode == 'master_doorkey':
        from train_master import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_second_step_doorkey.hocon')

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

    elif mode == 'ppo_dynobs':
        from train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_ppo_dynobst.hocon')

    elif mode == 'ppo_lava':
        from train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_ppo_lava.hocon')

    elif mode == 'ppo_doorkey':
        from train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_ppo_doorkey.hocon')

    elif mode == 'ppo_fourrooms':
        from train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_ppo_fourrooms.hocon')

    elif mode == 'ppo_keycorridor':
        from train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_ppo_keycorridor.hocon')

    elif mode == 'ppo_putnear':
        from train_direct_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_ppo_putnear.hocon')

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

    elif mode == 'rnd_ppo_keycorridor':
        from train_rnd_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_rnd_ppo_keycorridor.hocon')

    elif mode == 'rnd_ppo_putnear':
        from train_lang_rnd_ppo import gen_env_with_seed
        config = ConfigFactory.parse_file('conf/minigrid_rnd_ppo_putnear.hocon')

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
    parser.add_argument('--mode', choices=['worker', 'master', 'ssim_master', 'ssim_master_lava', 'ssim_worker',
                                           'worker_doorkey', 'master_doorkey',
                                           'direct_ppo', 'ppo_dynobs', 'ppo_lava', 'ppo_doorkey', 'ppo_fourrooms',
                                           'ppo_keycorridor', 'ppo_putnear',
                                           'rnd_ppo', 'rnd_ppo_lava','rnd_ppo_doorkey', 'rnd_ppo_fourrooms',
                                           'rnd_ppo_keycorridor', 'rnd_ppo_putnear',
                                           'multi'])
    parser.add_argument('--viz', action='store_true')
    parser.add_argument('--episodes', default=100, type=int)
    args = parser.parse_args()

    init_logger(__name__)
    rewards, steps, successes = play(args.mode, args.viz, args.episodes)
    logger.info(f'mode {args.mode}: '
                f'reward {rewards:.2f}, '
                f'steps {steps:.2f}, '
                f'success {successes:.2f}')
