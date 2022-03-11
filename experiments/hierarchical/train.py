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
UNK_GOAL = torch.zeros(GOAL_SIZE)


def bc_batch(rollouts, hindsight_goals, hindsight_returns, num_batches):
    T, B = rollouts.obs['image'].shape[:2]
    observations = rollouts.obs['image'].view(T * B, *rollouts.obs['image'].shape[2:])
    hindsight_goals = hindsight_goals.view(T * B, *hindsight_goals.shape[2:])
    returns = rollouts.returns.view(T * B, *rollouts.returns.shape[2:])
    hindsight_returns = hindsight_returns.view(T * B, *hindsight_returns.shape[2:])
    actions = rollouts.actions.view((T - 1) * B, *rollouts.actions.shape[2:])
    sil_ids = torch.where(hindsight_returns > returns)[0]

    observations = observations[sil_ids]
    hindsight_goals = hindsight_goals[sil_ids]
    actions = actions[sil_ids]
    for _ in range(num_batches):
        ids = torch.randint(0, len(sil_ids), (len(sil_ids) // num_batches,))
        yield {'image': observations[ids], 'goal': hindsight_goals[ids]}, actions[ids]


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

    # worker.actor_critic.enc.load_state_dict()
    # master = torch.load('artifacts/models/master.p', map_location='cpu')
    # test_vae(env, master)
    # exit(0)

    # tmp for check

    goals = []

    writer = SummaryWriter(config['outputs.logs'])

    worker_rollouts = RolloutStorage(
        config['training.n_steps'],
        config['training.n_processes'],
        gym.spaces.Dict({
            'image': env.observation_space,
            'goal': gym.spaces.Box(-np.inf, np.inf, (GOAL_SIZE,)),
        }),
        env.action_space
    )

    master_rollouts = RolloutStorage(
        config['training.n_steps'],
        config['training.n_processes'],
        env.observation_space,
        gym.spaces.Box(-np.inf, np.inf, (GOAL_SIZE,))
    )

    master_obs = env.reset()
    worker_obs = {
        'image': master_obs,
        'goal': master.act(master_obs)[1],
    }

    worker_rollouts.set_first_obs(worker_obs)
    worker_rollouts.to(config['agent.device'])

    master_rollouts.set_first_obs(master_obs)
    master_rollouts.to(config['agent.device'])

    # worker = torch.load('worker.p', map_location='cpu')
    # master = torch.load('master.p', map_location='cpu')

    def make_goal(img_obs):
        with torch.no_grad():
            return master.actor_critic.vae.encode(img_obs)

    def make_reward(img_obs, goal, thr=0.8):
        with torch.no_grad():
            obs_enc = master.actor_critic.vae.encode(img_obs)
            return (F.cosine_similarity(goal, obs_enc).unsqueeze(dim=1) > thr).float()

    start = time.time()
    num_updates = int(config['training.n_env_steps'] // config['training.n_steps'] // config['training.n_processes'])

    master_episode_rewards = deque(maxlen=10)
    worker_episode_rewards = deque(maxlen=10)

    for j in trange(num_updates):
        update_linear_schedule(worker.optimizer, j, num_updates, config['agent.lr'])
        update_linear_schedule(master.optimizer, j, num_updates, config['agent.lr'])

        WORKER_STEPS = 5

        for step in range(config['training.n_steps']):
            # Sample actions
            worker_value, worker_action, worker_action_log_prob, _ = worker.act(worker_obs)
            master_obs, master_reward, master_done, master_infos = env.step(worker_action)

            worker_reward = make_reward(master_obs, worker_obs['goal'])

            worker_timeout = torch.tensor(
                [info['steps'] % WORKER_STEPS == 0 for info in master_infos],
                device=worker_reward.device
            ).unsqueeze(dim=1)

            worker_done = \
                worker_timeout + \
                (worker_reward > 0) + \
                torch.tensor(master_done, device=worker_reward.device).unsqueeze(dim=1)

            master_value, master_action, master_action_log_prob, _ = master.act(master_obs)
            worker_obs = {
                'image': master_obs,
                'goal': master_action * worker_done + worker_obs['goal'] * ~worker_done,
            }

            for info in master_infos:
                if 'episode' in info.keys():
                    master_episode_rewards.append(info['episode']['r'])
            worker_episode_rewards.extend(worker_reward[torch.where(worker_done)[0]].view(-1))

            worker_rollouts.insert(
                obs=worker_obs,
                actions=worker_action,
                action_log_probs=worker_action_log_prob,
                value_preds=worker_value,
                rewards=worker_reward,
                masks=~worker_done
            )

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in master_done])

            master_rollouts.insert(
                obs=master_obs,
                actions=master_action,
                action_log_probs=master_action_log_prob,
                value_preds=master_value,
                rewards=master_reward,
                masks=masks)

        worker_next_value = worker.get_value(worker_rollouts.get_last_obs())
        worker_rollouts.compute_returns(worker_next_value, config['agent.gamma'], config['agent.gae_lambda'])

        hindsight_goals = torch.zeros_like(worker_rollouts.obs['goal'])
        hindsight_goals[-1] = worker_rollouts.obs['goal'][-1]
        hindsight_returns = torch.zeros_like(worker_rollouts.returns)
        for i in reversed(range(worker_rollouts.masks.size(0) - 1)):
            next_obs = worker_rollouts.obs['image'][i + 1]
            mask = worker_rollouts.masks[i + 1]
            master_mask = master_rollouts.masks[i + 1]
            hindsight_goals[i] = hindsight_goals[i + 1] * mask + make_goal(next_obs) * (1 - mask)

            # don't use hindsight goal if it is a new episode
            hindsight_goals[i] = hindsight_goals[i] * master_mask + worker_rollouts.obs['goal'][i] * (1.0 - master_mask)
            hindsight_returns[i] = \
                hindsight_returns[i + 1] * config['agent.gamma'] * mask + \
                make_reward(next_obs, hindsight_goals[i])

        #
        # from matplotlib import pyplot as plt
        # for i, (obs, h_goal, goal, mask, ret) in enumerate(zip(
        #         worker_rollouts.obs['image'],
        #         hindsight_goals,
        #         worker_rollouts.obs['goal'],
        #         worker_rollouts.masks,
        #         hindsight_returns
        # )):
        #     def dec(img):
        #         with torch.no_grad():
        #             return master.actor_critic.vae.decode(img)
        #
        #     env_id = 8
        #     fig, ax = plt.subplots(1, 3)
        #     plt.title(f'step {i}, done {1 - mask[env_id].item()}, return {ret[env_id].item()}')
        #     ax[0].imshow(obs[env_id].permute(1, 2, 0).int())
        #     ax[1].imshow(dec(h_goal)[env_id].permute(1, 2, 0))
        #     ax[2].imshow(dec(goal)[env_id].permute(1, 2, 0))
        #     plt.show()

        master_next_value = master.get_value(master_rollouts.get_last_obs())
        master_rollouts.compute_returns(master_next_value, config['agent.gamma'], config['agent.gae_lambda'])

        worker_value_loss, worker_action_loss, worker_dist_entropy = worker.update(worker_rollouts)
        master_value_loss, master_action_loss, master_dist_entropy, master_rec_loss, master_kl = \
            master.update(master_rollouts)

        worker_rollouts.after_update()
        master_rollouts.after_update()

        sil_worker_loss = 0
        sil_master_loss = 0

        for e in range(worker.ppo_epoch):
            for obs_batch, actions in bc_batch(worker_rollouts, hindsight_goals, hindsight_returns, worker.num_mini_batch):
                _, action_log_probs, _, _ = worker.actor_critic.evaluate_actions(obs_batch, actions, None, None)
                worker.optimizer.zero_grad()
                sil_loss = -torch.mean(action_log_probs)
                sil_loss.backward()
                worker.optimizer.step()
                sil_worker_loss += sil_loss.item()
        sil_worker_loss /= worker.ppo_epoch

        for e in range(master.ppo_epoch):
            for obs_batch, actions in bc_batch(worker_rollouts, hindsight_goals, hindsight_returns, master.num_mini_batch):
                _, action, _, _ = master.actor_critic.act(obs_batch['image'], obs_batch['goal'], None, None)
                master.optimizer.zero_grad()
                sil_loss = F.mse_loss(action, obs_batch['goal'])
                sil_loss.backward()
                master.optimizer.step()
                sil_master_loss += sil_loss.item()
        sil_master_loss /= master.ppo_epoch

        if j % config['training.verbose'] == 0 and len(master_episode_rewards) > 1:
            total_num_steps = (j + 1) * config['training.n_processes'] * config['training.n_steps']
            end = time.time()
            print(f'Updates {j}, '
                  f'num timesteps {total_num_steps}, '
                  f'FPS {int(total_num_steps / (end - start))} \n'
                  f'Last {len(master_episode_rewards)} training episodes: '
                  f'mean/median reward {np.mean(master_episode_rewards):.4f}/{np.median(master_episode_rewards):.4f}, '
                  f'min/max reward {np.min(master_episode_rewards):.4f}/{np.max(master_episode_rewards):.4f}\n'
                  f'worker_dist_entropy {worker_dist_entropy:.4f}, '
                  f'worker_value_loss {worker_value_loss:.4f}, '
                  f'worker_action_loss {worker_action_loss:.4f}\n'
                  f'master_dist_entropy {master_dist_entropy:.4f}, '
                  f'master_value_loss {master_value_loss:.4f}, '
                  f'master_action_loss {master_action_loss:.4f} '
                  f'master_rec_loss {master_rec_loss:.4f} '
                  f'master_kl_div {master_kl:.4f}\n'
                  f'sil_worker_loss {sil_worker_loss:.4f}, '
                  f'sil_master_loss {sil_master_loss:.4f}'
            )

            #writer.add_scalar('dist_entropy', dist_entropy, total_num_steps)
            #writer.add_scalar('value_loss', value_loss, total_num_steps)
            #writer.add_scalar('action_loss', action_loss, total_num_steps)
            #writer.add_scalar('reward', np.mean(episode_rewards), total_num_steps)

            torch.save(worker, 'worker.p')
            torch.save(master, 'master.p')
