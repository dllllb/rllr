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

    worker_policy = WorkerPolicyModel(env.observation_space.shape, env.action_space.n)
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

    #master_policy = PolicyModel(env.observation_space.shape, env.action_space.n)
    #master_policy.to(config['agent.device'])

    master = PPO(
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

    goals = []

    writer = SummaryWriter(config['outputs.logs'])

    worker_rollouts = RolloutStorage(
        config['training.n_steps'], config['training.n_processes'], env.observation_space, env.action_space
    )

    master_rollouts = RolloutStorage(
        config['training.n_steps'], config['training.n_processes'], env.observation_space, env.action_space
    )

    obs = env.reset()
    worker_rollouts.set_first_obs(obs)
    worker_rollouts.to(config['agent.device'])

    start = time.time()
    num_updates = int(config['training.n_env_steps'] // config['training.n_steps'] // config['training.n_processes'])

    episode_rewards = deque(maxlen=10)

    for j in trange(num_updates):
        update_linear_schedule(worker.optimizer, j, num_updates, config['agent.lr'])

        for step in range(config['training.n_steps']):
            # Sample actions
            value, action, action_log_prob, _ = worker.act(obs)
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            worker_rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        next_value = worker.get_value(worker_rollouts.get_last_obs())
        worker_rollouts.compute_returns(next_value, config['agent.gamma'], config['agent.gae_lambda'])

        value_loss, action_loss, dist_entropy = worker.update(worker_rollouts)
        worker_rollouts.after_update()

        if j % config['training.verbose'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * config['training.n_processes'] * config['training.n_steps']
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

            writer.add_scalar('dist_entropy', dist_entropy, total_num_steps)
            writer.add_scalar('value_loss', value_loss, total_num_steps)
            writer.add_scalar('action_loss', action_loss, total_num_steps)
            writer.add_scalar('reward', np.mean(episode_rewards), total_num_steps)

            torch.save(worker, config['outputs.model'])
