import logging
import os

from functools import partial

from gym_minigrid_navigation import environments as minigrid_envs
from gym_minigrid_navigation import encoders as minigrid_encoders
from dqn import get_dqn_agent
from models import get_master_worker_net
from rewards import get_reward_function
from utils import get_conf, init_logger, switch_reproducibility_on

logger = logging.getLogger(__name__)


def run_episode(env, agent, train_mode=True, max_steps=1_000):
    """
    A helper function for running single episode
    """

    state = env.reset()
    agent.explore = train_mode
    score, steps, done = 0, 0, False
    while not done and steps <= max_steps:
        steps += 1
        action = agent.act(state, env.goal_state)
        next_state, reward, done, _ = env.step(action)
        if train_mode and env.goal_state is not None:
            agent.update(state, env.goal_state, action, reward, next_state, done)

        score += reward
        state = next_state

    if train_mode:
        agent.reset_episode()
    env.close()

    return score, steps


def run_episodes(env, agent, n_episodes=1_000, train_mode=True, verbose=False, max_steps=100_000):
    """
    Runs a series of episode and collect statistics
    """
    score_sum, step_sum = 0, 0
    scores, steps = [], []
    for episode in range(1, n_episodes + 1):
        score, step = run_episode(env, agent, train_mode, max_steps)
        score_sum += score
        step_sum += step
        scores.append(score)
        steps.append(step)
        if verbose and episode % int(verbose) == 0:
            avg_score = score_sum / int(verbose)
            avg_steps = step_sum / int(verbose)
            logger.info(f"Episode: {episode}. scores: {avg_score:.2f}, steps: {avg_steps:.2f}")
            score_sum, step_sum, losses = 0, 0, []

    return scores, steps


def gen_env(conf, reward_function, verbose=False):
    if conf['env_type'] == 'gym_minigrid':
        env = minigrid_envs.gen_wrapped_env(conf, reward_function, verbose=verbose)
        return env
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")


def get_encoders(conf):
    if conf['env.env_type'] == 'gym_minigrid':
        state_encoder, goal_state_encoder = minigrid_encoders.get_encoders(conf)
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
    reward_function = get_reward_function(config)

    logger.info(f"Running agent training: {config['training.n_episodes']} episodes")
    env = gen_env(config['env'], reward_function)
    run_episodes(
        env=env,
        agent=agent,
        n_episodes=config['training.n_episodes'],
        verbose=config['training.verbose'],
        max_steps=config['training'].get('max_steps', 100_000),
    )

    if config.get('outputs', False):
        if config['outputs.save_example']:
            env = gen_env(config['env'], reward_function, verbose=True)
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
    init_logger('expected_steps')
    init_logger('gym_minigrid_navigation.environments')
    main()
