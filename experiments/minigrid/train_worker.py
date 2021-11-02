import logging
import os
import pickle
import torch

from functools import partial

import rllr.env as environments
from rllr.algo import PPO
from rllr.models.ppo import ActorCriticNetwork
from rllr.env.gym_minigrid_navigation import environments as minigrid_envs
from rllr.models import SameStatesCriterion, ActorNetwork, StateEmbedder, SSIMCriterion
from rllr.models.encoders import GoalStateEncoder
from rllr.models import encoders as minigrid_encoders
from rllr.utils import get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger
from rllr.utils import train_ppo
from rllr.env.vec_wrappers import make_vec_envs
from rllr.env.wrappers import EpisodeInfoWrapper


logger = logging.getLogger(__name__)


def get_goal_achieving_criterion(config):
    if config['goal_achieving_criterion'] == 'position_and_direction':
        def position_achievement(state, goal_state):
            return (state['position'] == goal_state['position']).all()
        return position_achievement
    elif config['goal_achieving_criterion'] == 'position':
        def position_achievement(state, goal_state):
            return (state['position'][:-1] == goal_state['position'][:-1]).all()
        return position_achievement
    elif config['goal_achieving_criterion'] == 'state_distance_network':
        encoder = torch.load(config['state_distance_network_params.path'], map_location='cpu')
        device = torch.device(config['state_distance_network_params.device'])
        threshold = config['state_distance_network_params.threshold']
        return SameStatesCriterion(encoder, device, threshold)
    elif config['goal_achieving_criterion'] == 'state_similarity':
        ssim = torch.load(config['ssim_network_params.path'])
        device = torch.device(config['ssim_network_params.device'])
        threshold = config['ssim_network_params.threshold']
        return SSIMCriterion(ssim, device, threshold)
    else:
        raise AttributeError(f"unknown goal_achieving_criterion '{config['env.goal_achieving_criterion']}'")


def gen_env(conf, verbose=False):
    if conf['env_type'] == 'gym_minigrid':
        env = minigrid_envs.gen_wrapped_env(conf, verbose=verbose)
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")
    return env


def gen_navigation_env(conf, env=None, verbose=False, goal_achieving_criterion=None):
    if not env:
        env = gen_env(conf=conf, verbose=verbose)

    if not goal_achieving_criterion:
        goal_achieving_criterion = get_goal_achieving_criterion(conf)

    if conf.get('state_distance_network_params', {}).get('path', False):
        encoder = torch.load(conf['state_distance_network_params.path'], map_location='cpu')
        device = torch.device(conf['state_distance_network_params.device'])
        embedder = StateEmbedder(encoder, device)
    else:
        embedder = None

    if conf.get('goal_type', None) == 'random':
        if conf['env_type'] == 'gym_minigrid':
            random_goal_generator = minigrid_envs.random_grid_goal_generator(conf, verbose=verbose)
        else:
            raise AttributeError(f"unknown env_type '{conf['env_type']}'")
    else:
        random_goal_generator = None

    env = environments.navigation_wrapper(
        env=env,
        conf=conf,
        goal_achieving_criterion=goal_achieving_criterion,
        random_goal_generator=random_goal_generator,
        verbose=verbose)

    if conf.get('intrinsic_episodic_reward', False):
        env = environments.IntrinsicEpisodicReward(env, embedder)

    if conf.get('random_network_distillation_reward', False):
        reward_conf = conf['random_network_distillation_reward']
        device = torch.device(reward_conf['device'])

        if conf['env_type'] == 'gym_minigrid':
            grid_size = conf['grid_size'] * conf.get('tile_size', 1)
            target_network = minigrid_encoders.get_encoder(grid_size, reward_conf['target'])
            predictor_network = minigrid_encoders.get_encoder(grid_size, reward_conf['predictor'])
        else:
            raise AttributeError(f"unknown env_type '{conf['env_type']}'")

        env = environments.RandomNetworkDistillationReward(env, target_network, predictor_network, device)

    return env


def get_encoders(conf):
    if conf['env.env_type'] == 'gym_minigrid':
        init_logger('rllr.env.gym_minigrid_navigation.environments')
        grid_size = conf['env.grid_size'] * conf['env'].get('tile_size', 1)
        state_encoder = minigrid_encoders.get_encoder(grid_size, conf['worker'])
        goal_state_encoder = minigrid_encoders.get_encoder(grid_size, conf['master'])
        return state_encoder, goal_state_encoder
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")


def get_hindsight_state_encoder(state_encoder, goal_state_encoder, config):
    hidden_size_worker = config['worker']['head.hidden_size']
    hidden_size_master = config['master']['head.hidden_size']
    action_size = config['env.action_size']
    emb_size = config['master']['emb_size']
    # FIXME: DDPG's actor looks strange here
    goal_state_encoder = ActorNetwork(emb_size, goal_state_encoder, hidden_size_master)
    return GoalStateEncoder(state_encoder=state_encoder, goal_state_encoder=goal_state_encoder)


def get_worker_agent(env, conf):
    hindsight_encoder = get_hindsight_state_encoder(*get_encoders(conf), conf)
    hidden_size = conf['worker.head.hidden_size']
    policy = ActorCriticNetwork(env.action_space, hindsight_encoder, hindsight_encoder, hidden_size, hidden_size)

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


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env.seed'] = seed
    return EpisodeInfoWrapper(gen_navigation_env(conf['env']))


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    agent = get_worker_agent(env, config)

    logger.info(f"Running agent training: { config['training.n_steps'] * config['training.n_processes']} steps")
    train_ppo(
        env=env,
        agent=agent,
        conf=config,
    )


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
