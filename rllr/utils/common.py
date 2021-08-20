import torch
import numpy as np
from tqdm import trange
import time
from collections import deque

from rllr.buffer.rollout import RolloutStorage


def switch_reproducibility_on(seed=42):
    import torch
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def convert_to_torch(arr, device='cpu'):
    if arr and isinstance(arr[0], dict):
        res = {
            key: convert_to_torch([x[key] for x in arr], device=device) for key in arr[0].keys()
        }
        return res

    else:
        res = np.vstack([np.expand_dims(x, axis=0) for x in arr])
        return torch.from_numpy(res).float().to(device)


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_ppo(env, agent, conf):
    """
    Runs a series of episode and collect statistics
    """
    rollouts = RolloutStorage(
        conf['training.n_steps'], conf['training.n_processes'], env.observation_space, env.action_space
    )
    obs = env.reset()
    rollouts.set_first_obs(obs)
    rollouts.to(conf['agent.device'])

    start = time.time()
    num_updates = int(conf['training.n_env_steps'] // conf['training.n_steps'] // conf['training.n_processes'])

    episode_rewards = deque(maxlen=10)

    for j in trange(num_updates):
        update_linear_schedule(agent.optimizer, j, num_updates, conf['agent.lr'])

        for step in range(conf['training.n_steps']):
            # Sample actions
            value, action, action_log_prob = agent.act(obs)
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        next_value = agent.get_value(rollouts.get_last_obs())
        rollouts.compute_returns(next_value, conf['agent.gamma'], conf['agent.gae_lambda'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)

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

            torch.save(agent, conf['outputs.path'])
