import logging
import torch

from rllr.algo import IMPPO
from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper, ZeroRewardWrapper
from rllr.models import encoders, ActorCriticNetwork, RNDModel, InverseDynamicsModel
from rllr.utils import im_train_ppo, get_conf, switch_reproducibility_on
from rllr.utils.inverse_dynamics_model import IDM
from rllr.utils.logger import init_logger

logger = logging.getLogger(__name__)


def gen_env(conf, verbose=False):
    if conf['env_type'] == 'gym_minigrid':
        env = minigrid_envs.gen_wrapped_env(conf, verbose=verbose)
    else:
        raise AttributeError(f"unknown env_type '{conf['env_type']}'")
    return env


def calculate_grid_size(config):
    if config['env.env_type'] == 'gym_minigrid':
        init_logger('rllr.env.gym_minigrid_navigation.environments')
        if config['env'].get('fully_observed', True):
            grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)
        else:
            grid_size = 7 * config['env'].get('tile_size', 1)
    else:
        raise AttributeError(f"unknown env_type '{config['env_type']}'")
    return grid_size


def get_agent(env, config):
    grid_size = calculate_grid_size(config)
    encoder = encoders.get_encoder(grid_size, config['worker'])
    hidden_size = config['worker.head.hidden_size']
    policy = ActorCriticNetwork(
        env.action_space, encoder,
        encoder, hidden_size,
        hidden_size, use_intrinsic_motivation=True,
        is_recurrent=True if config['worker.state_encoder_type'] == 'cnn_rnn' else False
    )

    rnd = RNDModel(
        encoders.get_encoder(grid_size, config['worker']),
        encoders.get_encoder(grid_size, config['worker']),
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


def get_idm(conf):
    grid_size = calculate_grid_size(conf)
    encoder = encoders.get_encoder(grid_size, conf['inverse_dynamics'])
    idm_network = InverseDynamicsModel(encoder,
                                       action_size=conf['env.action_size'],
                                       config=conf['inverse_dynamics'])
    idm = IDM(idm_network,
              lr=conf['inverse_dynamics.lr'],
              epochs=conf['inverse_dynamics.epochs'],
              verbose=conf['inverse_dynamics.verbose'])
    return idm


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed
    env = gen_env(conf['env'])

    env = ZeroRewardWrapper(env)
    env = EpisodeInfoWrapper(env)

    return env


def train_ssim_with_rnd(env, agent, idm, conf):
    def idm_update_callback(rollouts):
        return idm.update(rollouts)

    im_train_ppo(env, agent, conf, idm_update_callback)

    torch.save(idm, conf['outputs.path'])
    return agent, idm


def main(args=None):
    init_logger(__name__)
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    agent = get_agent(env, config)
    agent.to(config['agent.device'])
    idm = get_idm(config)

    logger.info(f"Running agent training: { config['training.n_steps'] * config['training.n_processes']} steps")
    train_ssim_with_rnd(
        env=env,
        agent=agent,
        idm=idm,
        conf=config
    )


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('gym_minigrid_navigation.environments')
    main()
