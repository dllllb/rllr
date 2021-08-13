import logging
import torch
from pathlib import Path

from experiments.train_state_distance_network import train_statedistance_network
from experiments.train_worker import get_worker_agent, train_worker, gen_navigation_env
from experiments.train_master import train_master, get_master_agent

from rllr.models.encoders import get_encoder
from rllr.models import EncoderDistance

from rllr.env.gym_minigrid_navigation.environments import gen_wrapped_env

from rllr.utils import get_conf


from rllr.utils import switch_reproducibility_on
from rllr.utils.logger import init_logger

logger = logging.getLogger(__name__)


def main(args=None):
    config = get_conf([f"-c{Path(__file__).parent.absolute()}/conf/experiments_minigrid_simple_cnn.hocon"])
    # config = get_conf(args)

    env = gen_wrapped_env(config['env'])

    grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)
    if "state_distance_encoder" in config:
        encoder = get_encoder(grid_size, config['state_distance_encoder'])

    device = torch.device(config['device'])

    worker_agent = get_worker_agent(config['worker_agent'])

    seeds = [0]
    for seed in seeds:
        switch_reproducibility_on(seed)

        if "state_distance_encoder" in config:
            logger.info(f"Running statedistance training: {config['state_distance_network']['training.n_episodes']} episodes")
            net = train_statedistance_network(config['state_distance_network'], encoder, env)
            threshold = config['state_distance_network']['state_distance_network_params.threshold']
            goal_achieving_criterion = EncoderDistance(net.encoder, device, threshold)
        else:
            goal_achieving_criterion = None

        # Train worker
        logger.info(f"Running worker agent training: {config['worker_agent']['training.n_episodes']} episodes")
        navigation_env = gen_navigation_env(config['worker_agent']['env'],
                                            goal_achieving_criterion=goal_achieving_criterion)
        train_worker(env=navigation_env, worker_agent=worker_agent,
                     n_episodes=config['worker_agent']['training']['n_episodes'],
                     verbose=config['worker_agent']['training']['verbose'],
                     max_steps=config['worker_agent']['training'].get('max_steps', 100_000),
                     )

        worker_agent.explore = False

        # Train master
        emb_size = worker_agent.qnetwork_local.state_encoder.goal_state_encoder.output_size
        master_agent = get_master_agent(emb_size, config['master_agent'])

        logger.info(f"Running master agent training: {config['master_agent.training.n_episodes']} episodes")
        train_master(
            env,
            worker_agent,
            master_agent,
            n_episodes=config['master_agent.training.n_episodes'],
            verbose=config['master_agent.training.verbose'],
            worker_steps=1
        )



if __name__ == '__main__':
        init_logger(__name__)
        init_logger('experiments.train_state_distance_network')
        init_logger('experiments.train_worker')
        init_logger('experiments.train_master')
        init_logger('rllr.env.wrappers')
        init_logger('rllr.env.gym_minigrid_navigation.environments')
        main()