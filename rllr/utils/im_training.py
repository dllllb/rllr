from tqdm import trange
import time
from collections import deque, defaultdict
import torch
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from rllr.buffer.im_rollout import IMRolloutStorage
from rllr.utils.common import RunningMeanStd


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_state(obs):
    if obs.__class__.__name__ == 'dict':
        return obs["state"]
    return obs


def init_obs_rms(env, conf):
    if env.observation_space.__class__.__name__ == 'Dict':
        state_shape = env.observation_space["state"].shape
    else:
        state_shape = env.observation_space.shape

    obs_rms = RunningMeanStd(shape=state_shape, device=conf['agent.device'])
    obs = env.reset()

    for _ in range(50):
        states = []
        for _ in trange(conf['training.n_steps']):
            action = torch.tensor([[env.action_space.sample()] for _ in range(get_state(obs).shape[0])])
            obs, _, _, _ = env.step(action)
            states.append(get_state(obs))

        obs_rms.update(torch.stack(states))
    return obs_rms


def im_train_ppo(env, agent, conf, after_epoch_callback=None):
    writer = SummaryWriter(conf['outputs.logs'])
    reward_rms = RunningMeanStd(device=conf['agent.device'])
    obs_rms = init_obs_rms(env, conf)

    # training starts
    rollouts = IMRolloutStorage(
        conf['training.n_steps'], conf['training.n_processes'], env.observation_space, env.action_space,
        conf.get('encoder.recurrent_hidden_size', 1)
    )

    obs = env.reset()
    rollouts.set_first_obs(obs)
    rollouts.to(conf['agent.device'])

    start = time.time()
    num_updates = int(conf['training.n_env_steps'] // conf['training.n_steps'] // conf['training.n_processes'])

    episode_stats = defaultdict(lambda: defaultdict(lambda: deque(maxlen=10)))

    for epoch in trange(num_updates):
        update_linear_schedule(agent.optimizer, epoch, num_updates, conf['agent.lr'])

        for step in range(conf['training.n_steps']):
            # Sample actions
            (value, im_value), action, action_log_prob, rnn_rhs = agent.act(
                obs,
                rollouts.recurrent_hidden_states[step],
                rollouts.masks[step]
            )

            obs, reward, done, infos = env.step(action)

            im_reward = agent.compute_intrinsic_reward(
                ((get_state(obs) - obs_rms.mean) / torch.sqrt(obs_rms.var)).clip(-5, 5))

            for info in infos:
                if 'episode' in info.keys():
                    task = info['episode'].get('task', 'unk')
                    for info_key, info_value in filter(lambda kv: kv[0] != 'task', info['episode'].items()):
                        episode_stats[task][info_key].append(info_value)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, rnn_rhs, action, action_log_prob, value, im_value, reward, im_reward, masks)

        im_ret = torch.zeros((conf['training.n_steps'] + 1, conf['training.n_processes']), device=conf['agent.device'])
        for i, rew in enumerate(reversed(rollouts.im_rewards)):
            k = conf['training.n_steps'] - i - 1
            im_ret[k] = conf['agent.im_gamma'] * im_ret[k + 1] + rew.view(-1)
        im_ret = im_ret[:-1]

        mean, var, count = torch.mean(im_ret), torch.var(im_ret), len(im_ret)
        reward_rms.update_from_moments(mean, var, count)

        obs_rms.update(get_state(rollouts.obs))

        rollouts.im_rewards /= torch.sqrt(reward_rms.var)

        next_value, next_im_value = agent.get_value(
            rollouts.get_last_obs(),
            rollouts.recurrent_hidden_states[-1],
            rollouts.masks[-1]
        )

        rollouts.compute_returns(next_value, conf['agent.gamma'], conf['agent.gae_lambda'])
        rollouts.compute_im_returns(next_im_value, conf['agent.im_gamma'], conf['agent.gae_lambda'])

        value_loss, im_value_loss, action_loss, dist_entropy = agent.update(rollouts, obs_rms)
        rollouts.after_update()

        if epoch % conf['training.verbose'] == 0:
            total_num_steps = (epoch + 1) * conf['training.n_processes'] * conf['training.n_steps']
            end = time.time()
            print(f'Updates {epoch}, '
                  f'num timesteps {total_num_steps}, '
                  f'FPS {int(total_num_steps / (end - start))} \n'
                  f'dist_entropy {dist_entropy:.2f}, '
                  f'value_loss {value_loss:.2f}, '
                  f'im_value_loss {im_value_loss:.2f}, '
                  f'action_loss {action_loss:.2f}'
            )

            writer.add_scalar('dist_entropy', dist_entropy, total_num_steps)
            writer.add_scalar('value_loss', value_loss, total_num_steps)
            writer.add_scalar('im_value_loss', im_value_loss, total_num_steps)
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

            if after_epoch_callback is not None:
                loss = after_epoch_callback(rollouts)
                print(f'loss: {loss:.2f}')
