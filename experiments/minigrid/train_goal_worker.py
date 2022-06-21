import logging
import torch

import rllr.env as environments
import rllr.models as models

from rllr.algo import PPO, IMPPO
from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper
from rllr.models import encoders
from rllr.utils import train_ppo, get_conf, switch_reproducibility_on, im_train_ppo
from rllr.utils.logger import init_logger


logger = logging.getLogger(__name__)


def gen_navigation_env(conf, verbose=True):
    env = minigrid_envs.gen_wrapped_env(conf, verbose=verbose)
    env = environments.TripletNavigationWrapper(env,
                                                choose_objects_uniformly=conf.get('choose_objects_uniformly', False))
    return env


def get_hindsight_state_encoder(env, config):
    grid_size = env.observation_space['state'].shape[0]
    if config['worker']['state_encoder_type'] == 'cnn_hindsight_encoder':
        return encoders.get_encoder(grid_size, config['worker'])
    else:
        state_encoder = encoders.get_encoder(grid_size, config['worker'])
        goal_state_encoder = models.MultiEmbeddingNetwork(config['goal_embedding'])
        return models.GoalStateEncoder(state_encoder=state_encoder, goal_state_encoder=goal_state_encoder)


def get_ppo_worker_agent(env, config):
    hindsight_encoder = get_hindsight_state_encoder(env, config)
    hidden_size = config['worker.head.hidden_size']
    policy = models.ActorCriticNetwork(env.action_space,
                                       hindsight_encoder,
                                       hindsight_encoder,
                                       hidden_size,
                                       hidden_size,
                                       same_encoders=True)

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


def get_imppo_worker_agent(env, config):
    grid_size = env.observation_space['state'].shape[0]
    hindsight_encoder = get_hindsight_state_encoder(env, config)
    hidden_size = config['worker.head.hidden_size']
    policy = models.ActorCriticNetwork(env.action_space,
                                       hindsight_encoder,
                                       hindsight_encoder,
                                       hidden_size,
                                       hidden_size,
                                       use_intrinsic_motivation=True,
                                       same_encoders=True)

    config['rnd']['input_size'] = policy.state_encoder.state_encoder.output_size
    rnd = models.RNDModel(encoders.get_encoder(grid_size, config['rnd']),
                          encoders.get_encoder(grid_size, config['rnd']),
                          config['agent.device'],
                          feature_extractor=policy.state_encoder.state_encoder)

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


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed
    return EpisodeInfoWrapper(gen_navigation_env(conf['env']))


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    if config['training.algorithm'] == 'ppo':
        agent = get_ppo_worker_agent(env, config)
        train = train_ppo
    elif config['training.algorithm'] == 'imppo':
        agent = get_imppo_worker_agent(env, config)
        train = im_train_ppo
    else:
        raise NotImplementedError(f'{config["training.algorithm"]} algorithm is not implemented')
    agent.to(config['agent.device'])

    logger.info(f"Running {config['training.algorithm']} agent training: {config['training.n_steps'] * config['training.n_processes']} steps")
    train(env=env,
          agent=agent,
          conf=config)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
