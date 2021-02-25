import logging
import numpy as np
import os

from copy import deepcopy
from functools import partial

from expected_steps import ExpectedStepsAmountLeaner
from gym_minigrid_navigation import environments as minigrid_envs
from gym_minigrid_navigation import encoders as minigrid_encoders
from dqn import get_dqn_agent
from models import get_master_worker_net
from rewards import get_reward_function, ExpectedStepsAmountReward
from utils import get_conf, init_logger, switch_reproducibility_on

logger = logging.getLogger(__name__)


def run_episode(env, agent, steps_learner=None, agent_train_mode=True, max_steps=1_000):
    """
    A helper function for running single episode
    """

    state = env.reset()
    agent.explore = agent_train_mode or steps_learner is not None

    buffer = [deepcopy(state)]

    score, steps, done = 0, 0, False
    while not done and steps <= max_steps:
        steps += 1
        action = agent.act(state, env.goal_state)
        next_state, reward, done, _ = env.step(action)
        if agent_train_mode:
            agent.update(state, env.goal_state, action, reward, next_state, done)

        buffer.append(deepcopy(state))

        score += reward
        state = next_state

    if steps_learner is not None:
        buffer = [(x, state, steps - i) for i, x in enumerate(buffer)]
        steps_learner.buffer.extend(buffer)
        loss = steps_learner.learn()
    else:
        loss = np.nan

    if agent_train_mode:
        agent.reset_episode()
    env.close()

    return score, steps, loss


def run_episodes(env, agent, steps_learner, n_episodes=1_000, agent_train_mode=True, verbose=False, max_steps=1_000):
    """
    Runs a series of episode and collect statistics
    """
    score_sum, step_sum = 0, 0
    scores, steps, losses = [], [], []
    for episode in range(1, n_episodes + 1):
        score, step, loss = run_episode(env, agent, steps_learner, agent_train_mode, max_steps)
        score_sum += score
        step_sum += step
        if loss is not None and not np.isnan(loss): losses.append(loss)
        scores.append(score)
        steps.append(step)
        if verbose and episode % int(verbose) == 0:
            avg_score = score_sum / int(verbose)
            avg_steps = step_sum / int(verbose)
            avg_loss = np.array(losses).mean() if losses else 0
            logger.info(f"Episode: {episode}. scores: {avg_score:.2f}, steps: {avg_steps:.2f}, L1loss: {avg_loss:.2f}")
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

    if config['training.reward'] != 'expected_steps_amount':
        reward_function = get_reward_function(config)
        expected_steps_learner = None
    else:
        # steps amount model trainings
        expected_steps_learner = ExpectedStepsAmountLeaner(config['expected_steps_params'])

        # warm-up
        logger.info(f"Running expected steps model warm-up: {config['expected_steps_params.warm_up']} episodes")
        if config.get('expected_steps_params.warm_up', 0):
            env = gen_env(config['env'], reward_function=lambda *args: 0)
            run_episodes(
                env=env,
                agent=agent,
                steps_learner=expected_steps_learner,
                n_episodes=config['expected_steps_params.warm_up'],
                agent_train_mode=False,
                verbose=config['training.verbose'],
                max_steps=config['expected_steps_params.warm_up_max_steps']
            )

        reward_function = ExpectedStepsAmountReward(expected_steps_learner.model)

    logger.info(f"Running agent training: {config['training.n_episodes']} episodes")
    env = gen_env(config['env'], reward_function)
    run_episodes(
        env=env,
        agent=agent,
        steps_learner=expected_steps_learner,
        n_episodes=config['training.n_episodes'],
        verbose=config['training.verbose']
    )

    if config.get('outputs', False):
        if config['outputs.save_example']:
            env = gen_env(config['env'], reward_function, verbose=True)
            logger.info(f"test episode: {run_episode(env, agent, agent_train_mode=False)}")
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
