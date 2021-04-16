import logging
import os
import pickle
import torch

from rllr.algo.ddpg import DDPGAgentMaster, MasterCriticNetwork
from train_worker import gen_env, get_encoders

from rllr.env.gym_minigrid_navigation import encoders as minigrid_encoders

from rllr.utils import get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger


logger = logging.getLogger(__name__)


def run_episode(env, worker_agent, master_agent):
    """
    A helper function for running single episode
    """
    state = env.reset()

    score, steps, done = 0, 0, False
    while not done:
        steps += 1
        goal_emb = master_agent.sample_goal(state)
        action = worker_agent.act({'state': state, 'goal_emb': goal_emb})
        next_state, reward, done, _ = env.step(action)
        score += reward
        master_agent.update(state, goal_emb, reward, next_state, done)
        state = next_state

    master_agent.reset_episode()
    env.close()

    return score, steps


def run_episodes(env, worker_agent, master_agent, n_episodes=1_000, verbose=False):
    """
    Runs a series of episode and collect statistics
    """
    score_sum, step_sum = 0, 0
    scores, steps = [], []
    for episode in range(1, n_episodes + 1):
        score, step = run_episode(env, worker_agent, master_agent)
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


def get_master_agent(emb_size, conf):
    if conf['env.env_type'] == 'gym_minigrid':
        grid_size = conf['env.grid_size'] * conf['env'].get('tile_size', 1)
        state_encoder = minigrid_encoders.get_encoder(grid_size, conf['master'])
        goal_state_encoder = minigrid_encoders.get_encoder(grid_size, conf['master'])
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")

    master_network = MasterCriticNetwork(emb_size, state_encoder, goal_state_encoder, conf['master'])
    master_agent = DDPGAgentMaster(master_network, conf['master_agent'])
    return master_agent


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = gen_env(config['env'])

    worker_agent = load_worker_agent(config['worker_agent'])
    emb_size = worker_agent.qnetwork_local.master.output_size

    master_agent = get_master_agent(emb_size, config)

    logger.info(f"Running agent training: {config['training.n_episodes']} episodes")
    run_episodes(
        env,
        worker_agent,
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
    init_logger('environments')
    init_logger('gym_minigrid_navigation.environments')
    main()
