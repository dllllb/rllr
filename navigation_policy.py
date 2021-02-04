import logging
import os

from functools import partial

from gym_minigrid_navigation import environments as minigrid_envs
from gym_minigrid_navigation import models as minigrid_models
from dqn import get_dqn_agent
from models import get_master_worker_net
from rewards import get_reward_function
from utils import get_conf, init_logger, switch_reproducibility_on

logger = logging.getLogger(__name__)


def run_episode(env, agent, train_mode=True):
    """
    A helper function for running single episode
    """

    state = env.reset()
    if not train_mode:
        agent.explore = False

    score, steps, done = 0, 0, False
    while not done:
        steps += 1
        action = agent.act(state, env.goal_state)
        next_state, reward, done, _ = env.step(action)
        if train_mode:
            agent.update(state, env.goal_state, action, reward, next_state, done)
        score += reward
        state = next_state

    agent.reset_episode()
    env.close()

    return score, steps


def run_episodes(env, agent, n_episodes=1_000, verbose=False):
    """
    Runs a series of episode and collect statistics
    """
    score_sum = 0
    scores = []
    steps = []
    for episode in range(1, n_episodes + 1):
        score, step = run_episode(env, agent, train_mode=True)
        score_sum += score
        scores.append(score)
        steps.append(step)
        if verbose and episode % int(verbose) == 0:
            avg_score = score_sum / int(verbose)
            logger.info("Episode: {}. Average score: {}".format(episode, avg_score))
            score_sum = 0

    return scores, steps


def gen_env(conf, reward_functions, verbose=False):
    if conf['env_type'] == 'gym_minigrid':
        env = minigrid_envs.gen_wrapped_env(conf, reward_functions, verbose=verbose)
        return env
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")


def get_encoders(conf):
    if conf['env.env_type'] == 'gym_minigrid':
        state_encoder, goal_state_encoder = minigrid_models.get_encoders(conf)
        return state_encoder, goal_state_encoder
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")


def get_agent(env, conf):
    if conf['agent.algorithm'] == 'DQN':
        state_encoder, goal_state_encoder = get_encoders(conf)
        get_net_function = partial(
            get_master_worker_net,
            state_encoder=state_encoder,
            goal_state_encoder=goal_state_encoder,
            action_size=env.action_size,
            config=conf
        )
        agent = get_dqn_agent(conf['agent'], get_net_function, env.action_size)
        return agent
    else:
        raise AttributeError(f"unknown algorithm '{conf['agent.algorithm']}'")


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    reward_functions = get_reward_function(config)
    env = gen_env(config['env'], reward_functions)
    agent = get_agent(env, config)

    run_episodes(env, agent, n_episodes=config['training.n_episodes'], verbose=config['training.verbose'])

    if config.get('outputs', False):
        if config['outputs.save_example']:
            env = gen_env(config['env'], reward_functions, verbose=True)
            logger.info(f"test episode: {run_episode(env, agent, train_mode=False)}")
            logger.info(f"episode's video saved")

        if config.get('outputs.path', False) and hasattr(agent, 'save_model'):

            save_dir = os.path.dirname(config['outputs.path'])
            os.makedirs(save_dir, exist_ok=True)

            agent.save_model(config['outputs.path'])
            logger.info(f"Models saved to '{config['outputs.path']}'")


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dqn')
    init_logger('gym_minigrid_navigation.environments')
    main()
