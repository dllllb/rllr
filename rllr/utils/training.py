from tqdm import trange
import time
from collections import deque, defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from rllr.buffer.rollout import RolloutStorage


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train_ppo(env, agent, conf):
    writer = SummaryWriter(conf['outputs.logs'])

    rollouts = RolloutStorage(
        conf['training.n_steps'], conf['training.n_processes'], env.observation_space, env.action_space,
        conf.get('worker.rnn_output', 32) * 2
    )
    obs = env.reset()
    rollouts.set_first_obs(obs)
    rollouts.to(conf['agent.device'])

    start = time.time()
    num_updates = int(conf['training.n_env_steps'] // conf['training.n_steps'] // conf['training.n_processes'])

    episode_stats = defaultdict(lambda: defaultdict(lambda: deque(maxlen=16)))

    for j in trange(num_updates):
        update_linear_schedule(agent.optimizer, j, num_updates, conf['agent.lr'])

        for step in range(conf['training.n_steps']):
            # Sample actions
            value, action, action_log_prob, rnn_rhs = agent.act(obs,
                                                                rollouts.recurrent_hidden_states[step],
                                                                rollouts.masks[step])
            obs, reward, done, infos = env.step(action)

            for info in infos:
                if 'episode' in info.keys():
                    task = info['episode'].get('task', 'unk')
                    for info_key, info_value in filter(lambda kv: kv[0] != 'task', info['episode'].items()):
                        episode_stats[task][info_key].append(info_value)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, rnn_rhs, action, action_log_prob, value, reward, masks)

        next_value = agent.get_value(rollouts.get_last_obs(),
                                     rollouts.recurrent_hidden_states[-1],
                                     rollouts.masks[-1])
        rollouts.compute_returns(next_value, conf['agent.gamma'], conf['agent.gae_lambda'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        rollouts.after_update()

        #if j % conf['training.verbose'] == 0 and len(episode_rewards) > 1:
        if j % conf['training.verbose'] == 0:
            total_num_steps = (j + 1) * conf['training.n_processes'] * conf['training.n_steps']
            end = time.time()
            print(f'Updates {j}, '
                  f'num timesteps {total_num_steps}, '
                  f'FPS {int(total_num_steps / (end - start))} \n'
                  f'dist_entropy {dist_entropy:.2f}, '
                  f'value_loss {value_loss:.2f}, '
                  f'action_loss {action_loss:.2f}')

            writer.add_scalar('dist_entropy', dist_entropy, total_num_steps)
            writer.add_scalar('value_loss', value_loss, total_num_steps)
            writer.add_scalar('action_loss', action_loss, total_num_steps)

            for task in episode_stats:
                print(f'Task {task}:')
                task_stats = episode_stats[task]
                for key, value in task_stats.items():
                    writer.add_scalar(f'{task}/{key}', np.mean(value), total_num_steps)
                    print(
                        f'mean/median {key} {np.mean(value):.2f}/{np.median(value):.2f}, '
                        f'min/max {key} {np.min(value):.2f}/{np.max(value):.2f}'
                    )
                print()

            torch.save(agent, conf['outputs.model'])
