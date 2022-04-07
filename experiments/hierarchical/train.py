import gym.spaces
import torch
import numpy as np
from collections import deque
from tqdm import trange
import time
from tensorboardX import SummaryWriter

from rllr.algo import PPO
from rllr.env import make_vec_envs
from rllr.utils import get_conf
from rllr.buffer.rollout import RolloutStorage

from worker import WorkerPolicyModel
from env import gen_env_with_seed
from utils import train_vae
from torch.nn import functional as F

from vae import VAE

import os


EMB_SIZE = 10


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


if __name__ == '__main__':
    config = get_conf()

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    vae = VAE(env.observation_space.shape, EMB_SIZE).to(config['agent.device'])

    worker_policy = WorkerPolicyModel(EMB_SIZE * 2, env.action_space.n)
    worker_policy.to(config['agent.device'])

    worker = PPO(
        worker_policy,
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm']
    )

    if not os.path.isfile('vae.pt'):
        train_vae(env, vae)
    else:
        vae.load_state_dict(
            torch.load(open('vae.pt', 'rb'), map_location=config['agent.device'])
        )

    writer = SummaryWriter(config['outputs.logs'])

    worker_rollouts = RolloutStorage(
        config['training.n_steps'],
        config['training.n_processes'],
        gym.spaces.Box(-np.inf, np.inf, (EMB_SIZE * 2,)),
        env.action_space
    )

    master_obs = env.reset()
    worker_obs = torch.cat([
        vae.encode(master_obs),
        torch.randn((master_obs.shape[0], EMB_SIZE,), device=config['agent.device'])
    ], dim=1)

    worker_rollouts.set_first_obs(worker_obs)
    worker_rollouts.to(config['agent.device'])

    start = time.time()
    num_updates = int(config['training.n_env_steps'] // config['training.n_steps'] // config['training.n_processes'])

    episode_rewards = deque(maxlen=10)

    for j in trange(num_updates):
        update_linear_schedule(worker.optimizer, j, num_updates, config['agent.lr'])

        for step in range(config['training.n_steps']):
            # Sample actions
            worker_value, worker_action, worker_action_log_prob, _ = worker.act(worker_obs)
            master_obs, master_reward, master_done, master_infos = env.step(worker_action)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in master_done])

            prev_goal = worker_obs[:, worker_obs.shape[1] // 2:]

            with torch.no_grad():
                master_obs_enc = vae.encode(master_obs)
                worker_reward = - (master_obs_enc - prev_goal).pow(2).sum(dim=1).unsqueeze(dim=1)

            print(worker_reward.mean())
            new_goal = torch.randn((prev_goal.shape[0], EMB_SIZE), device=prev_goal.device)
            goal = prev_goal * masks + new_goal * (1 - masks)
            worker_obs = torch.cat([master_obs_enc, goal], dim=1)

            for info in master_infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            worker_rollouts.insert(
                obs=worker_obs,
                actions=worker_action,
                action_log_probs=worker_action_log_prob,
                value_preds=worker_value,
                rewards=worker_reward,
                masks=masks
            )

        worker_next_value = worker.get_value(worker_rollouts.get_last_obs())
        worker_rollouts.compute_returns(worker_next_value, config['agent.gamma'], config['agent.gae_lambda'])

        worker_value_loss, worker_action_loss, worker_dist_entropy = worker.update(worker_rollouts)
        worker_rollouts.after_update()

        if j % config['training.verbose'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config['training.n_processes'] * config['training.n_steps']
            end = time.time()
            print(f'Updates {j}, '
                  f'num timesteps {total_num_steps}, '
                  f'FPS {int(total_num_steps / (end - start))} \n'
                  f'Last {len(episode_rewards)} training episodes: '
                  f'mean/median wreward {np.mean(episode_rewards):.4f}/{np.median(episode_rewards):.4f}, '
                  f'min/max wreward {np.min(episode_rewards):.4f}/{np.max(episode_rewards):.4f}\n'
                  f'worker_dist_entropy {worker_dist_entropy:.4f}, '
                  f'worker_value_loss {worker_value_loss:.4f}, '
                  f'worker_action_loss {worker_action_loss:.4f}'
            )

            torch.save(worker, 'worker.p')
