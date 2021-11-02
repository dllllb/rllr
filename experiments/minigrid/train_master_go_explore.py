import logging
import os
import pickle
import torch

from rllr.algo.go_explore import GoExplore

from experiments.train_worker import gen_env, get_goal_achieving_criterion

from rllr.utils import get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger


logger = logging.getLogger(__name__)


def run_episode(env, master_agent):
    """
    A helper function for running single episode
    """
    state = env.reset()
    score, steps, done = 0, 0, False
    while not done:
        go_reward, done, steps_go = master_agent.go(env, state)
        steps += steps_go
        score += go_reward
        if not done:
            explore_reward, done, steps_explore, state = master_agent.explore(env)
            steps += steps_explore
            score += explore_reward
    env.close()

    return score, steps


def run_episodes(env, master_agent, n_episodes=1_000, verbose=False):
    """
    Runs a series of episode and collect statistics
    """
    score_sum, step_sum = 0, 0
    scores, steps = [], []
    for episode in range(1, n_episodes + 1):
        score, step = run_episode(env, master_agent)
        score_sum += score
        step_sum += step
        scores.append(score)
        steps.append(step)

        if verbose and episode % int(verbose) == 0:
            avg_score = score_sum / int(verbose)
            avg_step = step_sum / int(verbose)
            logger.info(f"Episode: {episode}. scores: {avg_score:.2f}, steps: {avg_step:.2f}")
            score_sum, step_sum, goals_achieved_sum, losses = 0, 0, 0, []

    return scores, steps


def load_worker_agent(conf):
    with open(conf['path'], 'rb') as f:
        worker_agent = pickle.load(f)

    device = torch.device(conf['device'])
    worker_agent.qnetwork_local.to(device)
    return worker_agent


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])
    env = gen_env(config['env'])
    worker_agent = load_worker_agent(config['worker_agent'])
    worker_agent.explore = False

    goal_achieving_criterion = get_goal_achieving_criterion(config['env'])
    master_agent = GoExplore(worker_agent, goal_achieving_criterion=goal_achieving_criterion)

    logger.info(f"Running agent training: {config['training.n_episodes']} episodes")
    run_episodes(
        env,
        master_agent,
        n_episodes=config['training.n_episodes'],
        verbose=config['training.verbose']
    )

    if config.get('outputs', False):
        if config.get('outputs.path', False):
            save_dir = os.path.dirname(config['outputs.path'])
            os.makedirs(save_dir, exist_ok=True)

            with open(config['outputs.path'], 'wb') as f:
                pickle.dump(master_agent, f)

            logger.info(f"Master agent saved to '{config['outputs.path']}'")


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('dqn')
    init_logger('ddpg')
    init_logger('rllr.env.wrappers')
    init_logger('rllr.env.gym_minigrid_navigation.environments')
    main()
