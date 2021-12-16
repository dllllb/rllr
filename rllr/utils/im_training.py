from tqdm import trange
import time
from collections import deque
import torch
import numpy as np

from rllr.buffer.im_rollout import IMRolloutStorage
from rllr.utils.common import RunningMeanStd


def update_linear_schedule(optimizer, epoch, total_num_epochs, initial_lr):
    """Decreases the learning rate linearly"""
    lr = initial_lr - (initial_lr * (epoch / float(total_num_epochs)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def init_obs_rms(env, agent):
    obs_rms = RunningMeanStd(shape=env.observation_space.shape)
    states = []
    obs = env.reset()

    for j in trange(128):
        action = torch.tensor([[env.action_space.sample()] for _ in range(obs.shape[0])])
        obs, reward, done, infos = env.step(action)

        int_reward = agent.compute_intrinsic_reward(
            ((obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))

        states.append(obs)

    obs_rms.update(torch.stack(states))
    return obs_rms


def im_train_ppo(env, agent, conf):
    """
    Runs a series of episode and collect statistics
    """
    reward_rms = RunningMeanStd()
    obs_rms = init_obs_rms(env, agent)

    # training starts
    rollouts = IMRolloutStorage(
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
            (value, int_value), action, action_log_prob = agent.act(obs)
            obs, reward, done, infos = env.step(action)
            obs_rms.update(obs)

            int_reward = agent.compute_intrinsic_reward(
                ((obs - obs_rms.mean) / np.sqrt(obs_rms.var)).clip(-5, 5))

            for info in infos:
                if 'episode' in info.keys():
                    episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(obs, action, action_log_prob, value, int_value, reward, int_reward, masks)


        intret = torch.zeros((129, 16))
        for i, ret in enumerate(rollouts.int_rewards):
            intret[i + 1] = conf['agent.int_gamma'] * intret[i] + ret.view(-1)
        intret = intret[1:]

        mean, std, count = torch.mean(intret), torch.std(intret), len(intret)
        reward_rms.update_from_moments(mean, std ** 2, count)
        print(mean, std, count)

        obs_rms.update(rollouts.obs)

        rollouts.int_rewards /= torch.sqrt(reward_rms.var)
        next_value, next_int_value = agent.get_value(rollouts.get_last_obs())
        rollouts.compute_returns(next_value, conf['agent.gamma'], conf['agent.gae_lambda'])
        rollouts.compute_int_returns(next_int_value, conf['agent.int_gamma'], conf['agent.gae_lambda'])

        value_loss, int_value_loss, action_loss, dist_entropy = agent.update(rollouts, obs_rms)
        rollouts.after_update()


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
                  f'int_value_loss {int_value_loss:.2f}, '
                  f'action_loss {action_loss:.2f}'
                )

            torch.save(agent, conf['outputs.path'])
