import logging
import torch

import rllr.env as environments
import rllr.models as models

from rllr.algo import PPO, IMPPO
from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper
from rllr.models import encoders
from rllr.utils import train_ppo, im_train_ppo, get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger


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
        encoder.to(device)
        threshold = config['state_distance_network_params.threshold']
        return models.SameStatesCriterion(encoder, device, threshold)

    elif config['goal_achieving_criterion'] == 'state_similarity':
        ssim = torch.load(config['ssim_network_params.path'])
        device = torch.device(config['ssim_network_params.device'])
        threshold = config['ssim_network_params.threshold']
        return models.SSIMCriterion(ssim.ssim_network, device, threshold)
    else:
        raise AttributeError(f"unknown goal_achieving_criterion '{config['env.goal_achieving_criterion']}'")


def gen_env(conf, verbose=False):
    if conf['env_type'] == 'gym_minigrid':
        env = minigrid_envs.gen_wrapped_env(conf, verbose=verbose)
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")
    return env


def gen_navigation_env(conf, env=None, verbose=True, goal_achieving_criterion=None):
    if not env:
        env = gen_env(conf=conf, verbose=verbose)

    if not goal_achieving_criterion:
        goal_achieving_criterion = get_goal_achieving_criterion(conf)

    if conf.get('goal_type', None) == 'random':
        if conf['env_type'] == 'gym_minigrid':
            random_goal_generator = minigrid_envs.random_grid_goal_generator(conf, verbose=verbose)
        else:
            raise AttributeError(f"unknown env_type '{conf['env_type']}'")
    elif conf.get('goal_type', None) == 'walk':
        if conf['env_type'] == 'gym_minigrid':
            #random_goal_generator = minigrid_envs.walk_grid_goal_generator(conf, verbose=verbose)
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

    return env


def get_encoders(conf):
    if conf['env.env_type'] == 'gym_minigrid':
        init_logger('rllr.env.gym_minigrid_navigation.environments')
        if conf['env'].get('fully_observed', True):
            grid_size = conf['env.grid_size'] * conf['env'].get('tile_size', 1)
        else:
            grid_size = 7 * conf['env'].get('tile_size', 1)

        state_encoder = encoders.get_encoder(grid_size, conf['worker'])
        goal_state_encoder = encoders.get_encoder(grid_size, conf['master'])
        return state_encoder, goal_state_encoder
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")


def get_rnd_model(config):
    if config['env'].get('fully_observed', True):
        grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)
    else:
        grid_size = 7 * config['env'].get('tile_size', 1)
    rnd = models.RNDModel(
        encoders.get_encoder(grid_size, config['rnd']),
        encoders.get_encoder(grid_size, config['rnd']),
        config['agent.device'])

    return rnd


def get_imppo_worker_agent(env, config):
    if config['env.env_type'] == 'gym_minigrid':
        init_logger('rllr.env.gym_minigrid_navigation.environments')
        if config['env'].get('fully_observed', True):
            grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)
        else:
            grid_size = 7 * config['env'].get('tile_size', 1)

        state_encoder = encoders.get_encoder(grid_size, config['worker'])
    else:
        raise AttributeError(f"unknown env_type '{config['env_type']}'")

    hidden_size = config['worker.head.hidden_size']
    policy = models.ActorCriticNetwork(
        env.action_space, state_encoder,
        state_encoder, hidden_size,
        hidden_size, use_intrinsic_motivation=True
    )

    rnd = get_rnd_model(config)

    return IMPPO(
        policy,
        rnd,
        config['agent.ext_coef'],
        config['agent.im_coef'],
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm']
    )


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed
    return EpisodeInfoWrapper(gen_navigation_env(conf['env']))


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    if config['training.algorithm'] == 'imppo':
        agent = get_imppo_worker_agent(env, config)
        agent.to(config['agent.device'])

        steps = config['training.n_steps'] * config['training.n_processes']
        logger.info(f"Running {config['training.algorithm']} agent training: {steps} steps")
        im_train_ppo(
            env=env,
            agent=agent,
            conf=config,
        )

    else:
        raise AttributeError(f"unknown algorithm '{config['training.algorithm']}'")


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
