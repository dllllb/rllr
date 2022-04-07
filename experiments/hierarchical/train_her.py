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
from master import MasterPolicyModel, MasterPPO
from env import gen_env_with_seed
from utils import train_vae, test_vae
from torch.nn import functional as F

import os


GOAL_SIZE = 256


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

    worker_policy = WorkerPolicyModel(env.observation_space.shape, env.action_space.n, GOAL_SIZE)
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

    master_policy = MasterPolicyModel(env.observation_space.shape, GOAL_SIZE)
    master_policy.to(config['agent.device'])

    master = MasterPPO(
        master_policy,
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm']
    )

    if not os.path.isfile('master_agent.pt'):
        train_vae(env, master)
    else:
        master.actor_critic.load_state_dict(
            torch.load(open('master_agent.pt', 'rb'), map_location=config['agent.device'])
        )

    goals = []

    writer = SummaryWriter(config['outputs.logs'])

    worker_rollouts = RolloutStorage(
        config['training.n_steps'],
        config['training.n_processes'],
        gym.spaces.Dict({'image': env.observation_space, 'goal': gym.spaces.Box(-np.inf, np.inf, (GOAL_SIZE,))}),
        env.action_space
    )

    master_rollouts = RolloutStorage(
        config['training.n_steps'],
        config['training.n_processes'],
        env.observation_space,
        gym.spaces.Box(-np.inf, np.inf, (GOAL_SIZE,))
    )

    master_obs = env.reset()
    worker_obs = {'image': master_obs, 'goal': master.act(master_obs)[1]}
    worker_rollouts.set_first_obs(worker_obs)
    worker_rollouts.to(config['agent.device'])

    master_rollouts.set_first_obs(master_obs)
    master_rollouts.to(config['agent.device'])

    start = time.time()
    num_updates = int(config['training.n_env_steps'] // config['training.n_steps'] // config['training.n_processes'])

    episode_rewards = deque(maxlen=10)

    for j in trange(num_updates):
        update_linear_schedule(worker.optimizer, j, num_updates, config['agent.lr'])
        update_linear_schedule(master.optimizer, j, num_updates, config['agent.lr'])

        for step in range(config['training.n_steps']):
            # Sample actions
            prev_worker_obs = worker_obs
            worker_value, worker_action, worker_action_log_prob, _ = worker.act(worker_obs)
            master_obs, master_reward, master_done, master_infos = env.step(worker_action)

            master_value, master_action, master_action_log_prob, _ = master.act(master_obs)
            worker_obs = {'image': master_obs, 'goal': master_action}

            with torch.no_grad():
                worker_reward = F.cosine_similarity(
                    prev_worker_obs['goal'],
                    master.actor_critic.vae.encode(master_obs)
                ).unsqueeze(dim=-1)

            for info in master_infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in master_done])
            worker_rollouts.insert(worker_obs, worker_action, worker_action_log_prob, worker_value, worker_reward, masks)
            master_rollouts.insert(master_obs, master_action, master_action_log_prob, master_value, master_reward, masks)

        worker_next_value = worker.get_value(worker_rollouts.get_last_obs())
        worker_rollouts.compute_returns(worker_next_value, config['agent.gamma'], config['agent.gae_lambda'])

        master_next_value = master.get_value(master_rollouts.get_last_obs())
        master_rollouts.compute_returns(master_next_value, config['agent.gamma'], config['agent.gae_lambda'])

        worker_value_loss, worker_action_loss, worker_dist_entropy = worker.update(worker_rollouts)
        master_value_loss, master_action_loss, master_dist_entropy, master_rec_loss, master_kl = \
            master.update(master_rollouts)

        worker_rollouts.after_update()
        master_rollouts.after_update()

        if j % config['training.verbose'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config['training.n_processes'] * config['training.n_steps']
            end = time.time()
            print(f'Updates {j}, '
                  f'num timesteps {total_num_steps}, '
                  f'FPS {int(total_num_steps / (end - start))} \n'
                  f'Last {len(episode_rewards)} training episodes: '
                  f'mean/median reward {np.mean(episode_rewards):.2f}/{np.median(episode_rewards):.2f}, '
                  f'min/max reward {np.min(episode_rewards):.2f}/{np.max(episode_rewards):.2f}\n'
                  f'worker_dist_entropy {worker_dist_entropy:.2f}, '
                  f'worker_value_loss {worker_value_loss:.2f}, '
                  f'worker_action_loss {worker_action_loss:.2f}\n'
                  f'master_dist_entropy {master_dist_entropy:.2f}, '
                  f'master_value_loss {master_value_loss:.2f}, '
                  f'master_action_loss {master_action_loss:.2f} '
                  f'master_rec_loss {master_rec_loss:.2f} '
                  f'master_kl_div {master_kl:.2f}'
            )

            #writer.add_scalar('dist_entropy', dist_entropy, total_num_steps)
            #writer.add_scalar('value_loss', value_loss, total_num_steps)
            #writer.add_scalar('action_loss', action_loss, total_num_steps)
            #writer.add_scalar('reward', np.mean(episode_rewards), total_num_steps)

            torch.save(worker, 'worker.p')
            torch.save(master, 'master.p')
