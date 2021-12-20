from tqdm import trange
import time
from collections import deque, defaultdict
import torch
import numpy as np

from rllr.buffer.im_rollout import IMRolloutStorage
from rllr.utils.common import RunningMeanStd


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_obs_rms(env, conf):
    obs_rms = RunningMeanStd(shape=env.observation_space.shape, device=conf['agent.device'])
    states = []
    obs = env.reset()

    for j in trange(conf['training.n_steps']):
        action = torch.tensor([[env.action_space.sample()] for _ in range(obs.shape[0])])
        obs, reward, done, infos = env.step(action)
        states.append(obs)

    obs_rms.update(torch.stack(states))
    return obs_rms


def im_train_ppo(env, agent, conf, after_epoch_callback=None):
    """
    Runs a series of episode and collect statistics
    """
    reward_rms = RunningMeanStd(device=conf['agent.device'])
    obs_rms = init_obs_rms(env, conf)

    # training starts
    rollouts = IMRolloutStorage(
        conf['training.n_steps'], conf['training.n_processes'], env.observation_space, env.action_space,
        conf['encoder.recurrent_hidden_size']
    )

    obs = env.reset()
    rollouts.set_first_obs(obs)
    rollouts.to(conf['agent.device'])

    start = time.time()
    num_updates = int(conf['training.n_env_steps'] // conf['training.n_steps'] // conf['training.n_processes'])

    episode_rewards = defaultdict(lambda: deque(maxlen=10))
    episode_steps = defaultdict(lambda: deque(maxlen=10))
    episode_rooms = defaultdict(lambda: deque(maxlen=10))

    for j in trange(num_updates):
        update_linear_schedule(agent.optimizer, j, num_updates, conf['agent.lr'])

        for step in range(conf['training.n_steps']):
            # Sample actions
            (value, im_value), action, action_log_prob, rnn_rhs = agent.act(
                rollouts.obs[step],
                rollouts.recurrent_hidden_states[step],
                rollouts.masks[step]
            )

            obs, reward, done, infos = env.step(action)
            obs_rms.update(obs)

            im_reward = agent.compute_intrinsic_reward(
                ((obs - obs_rms.mean) / torch.sqrt(obs_rms.var)).clip(-5, 5))

            for info in infos:
                if 'episode' in info.keys():
                    task = info['episode'].get('task', 'unk')
                    episode_rewards[task].append(info['episode']['r'])
                    episode_steps[task].append(info['episode']['steps'])
                    episode_rooms[task].append(info['episode']['visited_rooms'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, rnn_rhs, action, action_log_prob, value, im_value, reward, im_reward, masks)

        im_ret = torch.zeros((conf['training.n_steps'] + 1, conf['training.n_processes']), device=conf['agent.device'])
        for i, rew in enumerate(rollouts.im_rewards):
            im_ret[i + 1] = conf['agent.im_gamma'] * im_ret[i] + rew.view(-1)
        im_ret = im_ret[1:]

        mean, var, count = torch.mean(im_ret), torch.var(im_ret), len(im_ret)
        reward_rms.update_from_moments(mean, var, count)

        obs_rms.update(rollouts.obs)

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

        if j % conf['training.verbose'] == 0:
            total_num_steps = (j + 1) * conf['training.n_processes'] * conf['training.n_steps']
            end = time.time()
            print(f'Updates {j}, '
                  f'num timesteps {total_num_steps}, '
                  f'FPS {int(total_num_steps / (end - start))} \n'
                  f'dist_entropy {dist_entropy:.2f}, '
                  f'value_loss {value_loss:.2f}, '
                  f'im_value_loss {im_value_loss:.2f}, '
                  f'action_loss {action_loss:.2f}'
                )
            for task in episode_rewards:
                print(
                    f'Task {task}, last {len(episode_rewards[task])} training episodes:\n'                    
                    
                    f'mean/median reward {np.mean(episode_rewards[task]):.2f}/{np.median(episode_rewards[task]):.2f}, '
                    f'min/max reward {np.min(episode_rewards[task]):.2f}/{np.max(episode_rewards[task]):.2f}\n'
                   
                    f'mean/median steps {np.mean(episode_steps[task]):.2f}/{np.median(episode_steps[task]):.2f}, '
                    f'min/max steps {np.min(episode_steps[task]):.2f}/{np.max(episode_steps[task]):.2f}\n'

                    f'mean/median rooms {np.mean(episode_rooms[task]):.2f}/{np.median(episode_rooms[task]):.2f}, '
                    f'min/max rooms {np.min(episode_rooms[task]):.2f}/{np.max(episode_rooms[task]):.2f}\n'
                )

            torch.save(agent, conf['outputs.path'])

        if after_epoch_callback is not None:
            loss = after_epoch_callback(rollouts)
            if j % conf['training.verbose'] == 0 and len(episode_rewards) > 1:
                print(f'loss: {loss:.2f}')
