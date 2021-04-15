import logging
import os
import pickle
import torch

from functools import partial

import environments
from gym_minigrid_navigation import environments as minigrid_envs
from gym_minigrid_navigation import encoders as minigrid_encoders
from dqn import get_dqn_agent
from models import get_master_worker_net, EncoderDistance
from utils import get_conf, init_logger, switch_reproducibility_on

logger = logging.getLogger(__name__)


def run_episode(env, worker_agent, train_mode=True, max_steps=1_000):
    """
    A helper function for running single episode
    """

    state = env.reset()
    worker_agent.explore = train_mode

    score, steps, done = 0, 0, False
    while not done and steps < max_steps:
        steps += 1
        action = worker_agent.act(state, env.goal_state)
        next_state, reward, done, _ = env.step(action)
        if train_mode and env.goal_state is not None:
            worker_agent.update(state, env.goal_state, action, reward, next_state, done)
        score += reward
        state = next_state

    if train_mode and env.goal_state is not None:
        worker_agent.reset_episode()

    env.close()
    return score, steps, env.is_goal_achieved


def run_episodes(env, worker_agent, n_episodes=1_000, verbose=False, train_mode=True, max_steps=256):
    """
    Runs a series of episode and collect statistics
    """
    score_sum, step_sum, goals_achieved_sum = 0, 0, 0
    scores, steps = [], []
    for episode in range(1, n_episodes + 1):
        score, step, goal_achieved = run_episode(env, worker_agent, train_mode, max_steps)
        score_sum += score
        step_sum += step
        goals_achieved_sum += goal_achieved
        scores.append(score)
        steps.append(step)

        if verbose and episode % int(verbose) == 0:
            avg_score = score_sum / int(verbose)
            avg_step = step_sum / int(verbose)
            avg_goal = goals_achieved_sum / int(verbose)
            logger.info(f"Episode: {episode}. scores: {avg_score:.2f}, steps: {avg_step:.2f}, achieved: {avg_goal:.2f}")
            score_sum, step_sum, goals_achieved_sum = 0, 0, 0

    return scores, steps


def get_goal_achieving_criterion(config):
    if config['goal_achieving_criterion'] == 'position':
        return lambda state, goal_state: (state['position'] == goal_state['position']).all()
    elif config['goal_achieving_criterion'] == 'state_distance_network':
        encoder = torch.load(config['state_distance_network_params.path'])
        device = torch.device(config['state_distance_network_params.device'])
        threshold = config['state_distance_network_params.threshold']
        return EncoderDistance(encoder, device, threshold)
    else:
        raise AttributeError(f"unknown goal_achieving_criterion '{config['env.goal_achieving_criterion']}'")


def gen_env(conf, verbose=False):
    if conf['env_type'] == 'gym_minigrid':
        env = minigrid_envs.gen_wrapped_env(conf, verbose=verbose)
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")
    return env


def gen_navigation_env(conf, verbose=False):
    env = gen_env(conf=conf, verbose=verbose)
    goal_achieving_criterion = get_goal_achieving_criterion(conf)

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
    return env


def get_encoders(conf):
    if conf['env.env_type'] == 'gym_minigrid':
        grid_size = conf['env.grid_size'] * conf['env'].get('tile_size', 1)
        state_encoder = minigrid_encoders.get_encoder(grid_size, conf['master'])
        goal_state_encoder = minigrid_encoders.get_encoder(grid_size, conf['master'])
        return state_encoder, goal_state_encoder
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")


def get_agent(conf):
    if conf['agent.algorithm'] == 'DQN':
        state_encoder, goal_state_encoder = get_encoders(conf)
        get_net_function = partial(
            get_master_worker_net,
            state_encoder=state_encoder,
            goal_state_encoder=goal_state_encoder,
            action_size=conf['env.action_size'],
            config=conf
        )
        agent = get_dqn_agent(conf['agent'], get_net_function, conf['env.action_size'])
        return agent
    else:
        raise AttributeError(f"unknown algorithm '{conf['agent.algorithm']}'")


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    agent = get_agent(config)
    env = gen_navigation_env(config['env'])

    logger.info(f"Running agent training: {config['training.n_episodes']} episodes")
    run_episodes(
        env=env,
        worker_agent=agent,
        n_episodes=config['training.n_episodes'],
        verbose=config['training.verbose'],
        max_steps=config['training'].get('max_steps', 100_000),
    )

    if config['training'].get('validation', False):
        logger.info(f"Running validation: {100} episodes")
        # validation on random goal w/o train_mode
        config['env']['goal_type'] = 'random'
        env = gen_navigation_env(config['env'])
        run_episodes(
            env=env,
            worker_agent=agent,
            train_mode=False,
            n_episodes=100,
            verbose=config['training.verbose']
        )

    if config.get('outputs', False):
        if config['outputs.save_example']:
            env = gen_navigation_env(config['env'], verbose=True)
            env = environments.visualisation_wrapper(env, config['env.video_path'])
            logger.info(f"test episode: {run_episode(env, agent, train_mode=False)}")
            logger.info(f"episode's video saved")

        if config.get('outputs.path', False):

            save_dir = os.path.dirname(config['outputs.path'])
            os.makedirs(save_dir, exist_ok=True)

            with open(config['outputs.path'], 'wb') as f:
                pickle.dump(agent, f)

            logger.info(f"Agent saved to '{config['outputs.path']}'")


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dqn')
    init_logger('environments')
    init_logger('gym_minigrid_navigation.environments')
    main()
