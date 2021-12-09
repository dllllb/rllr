import logging
import torch

from experiments.minigrid.train_worker import rnd_wrapper
from rllr.env.gym_minigrid_navigation import environments as minigrid_envs
from rllr.utils import train_ppo
from rllr.env.vec_wrappers import make_vec_envs
from rllr.utils import switch_reproducibility_on, get_conf
from rllr.env.wrappers import EpisodeInfoWrapper
from rllr.models import encoders
from rllr.models.ppo import ActorCriticNetwork
from rllr.algo import PPO
from rllr.utils.logger import init_logger
import rllr.env as environments

logger = logging.getLogger(__name__)


def gen_wrapped_env(conf):
    env = minigrid_envs.gen_wrapped_env(conf)

    if conf.get('random_network_distillation_reward', False):
        env = rnd_wrapper(env, conf)

    return env


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed
    return EpisodeInfoWrapper(gen_wrapped_env(conf['env']))


def get_agent(env, config):
    state_conf = config['encoder']
    hidden_size = state_conf['head']['hidden_size']
    grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)
    state_encoder = encoders.get_encoder(grid_size, config['encoder'])
    policy = ActorCriticNetwork(env.action_space, state_encoder, state_encoder, hidden_size, hidden_size)

    return PPO(
        policy,
        config['agent.clip_param'],
        config['agent.ppo_epoch'],
        config['agent.num_mini_batch'],
        config['agent.value_loss_coef'],
        config['agent.entropy_coef'],
        config['agent.lr'],
        config['agent.eps'],
        config['agent.max_grad_norm']
    )


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    agent = get_agent(env, config)
    agent.to(config['agent.device'])

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
