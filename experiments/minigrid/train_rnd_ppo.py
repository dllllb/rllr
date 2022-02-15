import logging

from rllr.algo import IMPPO
from rllr.env import EpisodeInfoWrapper, make_vec_envs, minigrid_envs
from rllr.models import encoders, ActorCriticNetwork, RNDModel
from rllr.utils import im_train_ppo
from rllr.utils import switch_reproducibility_on, get_conf
from rllr.utils.logger import init_logger

logger = logging.getLogger(__name__)


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed
    env = minigrid_envs.gen_wrapped_env(conf['env'])
    return EpisodeInfoWrapper(env)


def get_agent(env, config):
    state_conf = config['encoder']
    hidden_size = state_conf['head']['hidden_size']
    grid_size = env.observation_space.shape[0]
    state_encoder = encoders.get_encoder(grid_size, config['encoder'])
    policy = ActorCriticNetwork(
        env.action_space, state_encoder,
        state_encoder, hidden_size,
        hidden_size, use_intrinsic_motivation=True
    )

    rnd = RNDModel(
        encoders.get_encoder(grid_size, config['encoder']),
        encoders.get_encoder(grid_size, config['encoder']),
        config['agent.device'])

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
    im_train_ppo(
        env=env,
        agent=agent,
        conf=config,
    )


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
