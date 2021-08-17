import logging
import os
import pickle
import torch

from rllr.algo import PPO
from rllr.models.ppo import ActorCriticNetwork
from experiments.train_worker import gen_env
from rllr.env.vec_wrappers import make_vec_envs
from rllr.buffer.rollout import RolloutStorage
from rllr.env.wrappers import HierarchicalWrapper

from rllr.models import encoders as minigrid_encoders

from rllr.utils import get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger
from tqdm import trange

from collections import deque
import time
from tqdm import trange
import numpy as np


logger = logging.getLogger(__name__)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_master(env, master_agent, conf):
    """
    Runs a series of episode and collect statistics
    """
    rollouts = RolloutStorage(
        conf['training.n_steps'], conf['training.n_processes'], env.observation_space.shape, env.action_space.shape[0]
    )
    obs = env.reset()
    rollouts.obs[0].copy_(obs)
    rollouts.to(conf['agent.device'])

    start = time.time()
    num_updates = int(conf['training.n_env_steps'] // conf['training.n_steps'] // conf['training.n_processes'])

    episode_rewards = deque(maxlen=10)


    for j in trange(num_updates):
        update_linear_schedule(master_agent.optimizer, j, num_updates, conf['agent.lr'])

        for step in range(conf['training.n_steps']):
            # Sample actions
            value, action, action_log_prob = master_agent.act(obs)
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        next_value = master_agent.get_value(rollouts.obs[-1])
        rollouts.compute_returns(next_value, conf['agent.gamma'], conf['agent.gae_lambda'])

        value_loss, action_loss, dist_entropy = master_agent.update(rollouts)

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


def load_worker_agent(conf):
    with open(conf['path'], 'rb') as f:
        worker_agent = pickle.load(f)

    device = torch.device(conf['device'])
    worker_agent.qnetwork_local.to(device)
    return worker_agent


def get_master_agent(emb_size, conf):
    if conf['env.env_type'] == 'gym_minigrid':
        grid_size = conf['env.grid_size'] * conf['env'].get('tile_size', 1)
        state_encoder = minigrid_encoders.get_encoder(grid_size, conf['master'])
        goal_state_encoder = minigrid_encoders.get_encoder(grid_size, conf['master'])
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")

    hidden_size = conf['master.head.hidden_size']
    master_network = ActorCriticNetwork(emb_size, state_encoder, goal_state_encoder,
                                 actor_hidden_size=hidden_size, critic_hidden_size=hidden_size)

    master_agent = PPO(
        master_network,
        conf['agent.clip_param'],
        conf['agent.ppo_epoch'],
        conf['agent.num_mini_batch'],
        conf['agent.value_loss_coef'],
        conf['agent.entropy_coef'],
        conf['agent.lr'],
        conf['agent.eps'],
        conf['agent.max_grad_norm']
    )
    return master_agent


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env.seed'] = seed

    worker_agent = load_worker_agent(conf['worker_agent'])
    worker_agent.explore = False
    emb_size = worker_agent.qnetwork_local.state_encoder.goal_state_encoder.output_size
    return HierarchicalWrapper(gen_env(conf['env']), worker_agent, (emb_size,), n_steps=1)


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['env.num_processes'],
        config['agent.device']
    )

    master_agent = get_master_agent(env.action_space.shape[0], config)

    logger.info(f"Running agent training: { config['training.n_steps'] * config['training.n_processes']} episodes")
    train_master(env, master_agent, config)

    if config.get('outputs', False):
        if config.get('outputs.path', False):
            save_dir = os.path.dirname(config['outputs.path'])
            os.makedirs(save_dir, exist_ok=True)

            with open(config['outputs.path'], 'wb') as f:
                pickle.dump(master_agent, f)

            logger.info(f"Master agent saved to '{config['outputs.path']}'")


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dqn')
    init_logger('ppo')
    init_logger('rllr.env.wrappers')
    init_logger('rllr.env.gym_minigrid_navigation.environments')
    main()
