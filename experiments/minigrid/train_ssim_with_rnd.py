import logging
import numpy as np
import torch
from tqdm import trange
import time
from collections import deque, defaultdict
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


from rllr.algo import PPO
from rllr.buffer.rollout import RolloutStorage
from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper, RandomNetworkDistillationReward
from rllr.models import encoders, ActorCriticNetwork, StateSimilarityNetwork
from rllr.utils import get_conf, switch_reproducibility_on
from rllr.utils.state_similarity import ContrastiveStateSimilarity
from rllr.utils.logger import init_logger
from rllr.utils.training import update_linear_schedule

logger = logging.getLogger(__name__)


def gen_env(conf, verbose=False):
    if conf['env_type'] == 'gym_minigrid':
        env = minigrid_envs.gen_wrapped_env(conf, verbose=verbose)
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")
    return env


def get_agent(env, conf):
    grid_size = conf['env.grid_size'] * conf['env'].get('tile_size', 1)
    encoder = encoders.get_encoder(grid_size, conf['worker'])
    hidden_size = conf['worker.head.hidden_size']
    policy = ActorCriticNetwork(env.action_space, encoder, encoder, hidden_size, hidden_size)

    return PPO(
        policy,
        conf['agent.clip_param'],
        conf['agent.ppo_epoch'],
        conf['agent.num_mini_batch'],
        conf['agent.value_loss_coef'],
        conf['agent.entropy_coef'],
        conf['agent.lr'],
        conf['agent.eps'],
        conf['agent.max_grad_norm']
    )


def get_ssim(conf):
    grid_size = conf['env.grid_size'] * conf['env'].get('tile_size', 1)
    encoder = encoders.get_encoder(grid_size, conf['state_similarity'])
    ssim_network = StateSimilarityNetwork(encoder, conf['state_similarity.hidden_size'])
    ssim = ContrastiveStateSimilarity(ssim_network,
                                      lr=conf['state_similarity.lr'],
                                      radius=conf['state_similarity.radius'],
                                      n_updates=conf['state_similarity.n_updates'],
                                      epochs=conf['state_similarity.epochs'],
                                      verbose=conf['state_similarity.verbose'])
    return ssim


def gen_env_with_seed(conf, seed):

    conf['env.deterministic'] = True
    conf['env']['seed'] = seed
    env = gen_env(conf['env'])

    reward_conf = conf['random_network_distillation_reward']
    device = torch.device(reward_conf['device'])

    if conf['env.env_type'] == 'gym_minigrid':
        grid_size = conf['env.grid_size'] * conf.get('env.tile_size', 1)
        target_network = encoders.get_encoder(grid_size, reward_conf['target'])
        predictor_network = encoders.get_encoder(grid_size, reward_conf['predictor'])
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")

    env = RandomNetworkDistillationReward(env, target_network, predictor_network, device)
    return EpisodeInfoWrapper(env)


def show_visits(visit_stats, action_stats):
    indx = set([k[0] for k in visit_stats])
    cols = set([k[1] for k in visit_stats])
    df_visits = pd.DataFrame(index=indx, columns=cols)

    for k in visit_stats:
        df_visits.loc[k[0], k[1]] = visit_stats[k]
    df_visits.fillna(0, inplace=True)
    logger.info(df_visits)

    sns.heatmap(df_visits, annot=False)
    plt.show()

    logger.info("Actions stats")
    a_sum = sum([action_stats[a] for a in  action_stats])
    for a in action_stats:
        logger.info(f'{a}: {action_stats[a]/a_sum}')


def train_ssim_with_rnd(env, agent, ssim, conf):
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
    logger.info(f"Total number of updates: {num_updates}.")

    episode_rewards = deque(maxlen=10)
    visit_stats = defaultdict(int)
    action_stats = defaultdict(int)

    for j in trange(num_updates):
        update_linear_schedule(agent.optimizer, j, num_updates, conf['agent.lr'])

        for step in range(conf['training.n_steps']):
            # Sample actions
            value, action, action_log_prob = agent.act(obs)

            for a in action.squeeze(-1).numpy():
                action_stats[a] += 1

            obs, reward, dones, infos = env.step(action)
            for info, done in zip(infos, dones):
                if done:
                    for k in info['visit_stats']:
                        visit_stats[k] += info['visit_stats'][k]

                    if 'episode' in info.keys():
                        episode_rewards.append(info['episode']['r'])

            # If done then clean the history of observations.
            masks = torch.FloatTensor(dones)
            rollouts.insert(obs, action, action_log_prob, value, reward, masks)

        next_value = agent.get_value(rollouts.get_last_obs())
        rollouts.compute_returns(next_value, conf['agent.gamma'], conf['agent.gae_lambda'])

        value_loss, action_loss, dist_entropy = agent.update(rollouts)
        ssim_loss = ssim.update(rollouts)

        if j % conf['training.verbose'] == 0 and len(episode_rewards) > 1:
            total_num_steps = (j + 1) * conf['training.n_processes'] * conf['training.n_steps']
            end = time.time()
            logger.info(f'Updates {j}, '
                        f'num timesteps {total_num_steps}, '
                        f'FPS {int(total_num_steps / (end - start))} \n'
                        f'Last {len(episode_rewards)} training episodes: '
                        f'mean/median reward {np.mean(episode_rewards):.2f}/{np.median(episode_rewards):.2f}, '
                        f'min/max reward {np.min(episode_rewards):.2f}/{np.max(episode_rewards):.2f}\n'
                        f'dist_entropy {dist_entropy:.2f}, '
                        f'value_loss {value_loss:.2f}, '
                        f'action_loss {action_loss:.2f}, '
                        f'ssim_loss {ssim_loss:.6f}')

    torch.save(ssim, conf['outputs.path'])
    show_visits(visit_stats, action_stats)

    return agent, ssim


def main(args=None):
    init_logger("experiments.train_ssim_with_rnd")
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    agent = get_agent(env, config)
    ssim = get_ssim(config)

    logger.info(f"Running agent training: { config['training.n_steps'] * config['training.n_processes']} steps")
    train_ssim_with_rnd(
        env=env,
        agent=agent,
        ssim=ssim,
        conf=config
    )


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('gym_minigrid_navigation.environments')
    main()
