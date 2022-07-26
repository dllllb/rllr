import logging
import torch

import rllr.env as environments
import rllr.models as models

from rllr.algo import PPO, IMPPO
from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper
from rllr.models import encoders
from rllr.utils import train_ppo, im_train_ppo, get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger
from vae import VAE
from torch import nn


logger = logging.getLogger(__name__)


class VAEEncoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.vae.encode_and_sample(x)
            #x = self.vae.encode(x)
        return x


def get_goal_achieving_criterion(config):
    if config['goal_achieving_criterion'] == 'position_and_direction':
        def position_achievement(state, goal_state):
            return (state['position'] == goal_state['position']).all()
        return position_achievement

    elif config['goal_achieving_criterion'] == 'position':
        def position_achievement(state, goal_state):
            return (state['position'][:-1] == goal_state['position'][:-1]).all()
        return position_achievement

    elif config['goal_achieving_criterion'] == 'state_distance_network':
        encoder = torch.load(config['state_distance_network_params.path'], map_location='cpu')
        device = torch.device(config['state_distance_network_params.device'])
        encoder.to(device)
        threshold = config['state_distance_network_params.threshold']
        return models.SameStatesCriterion(encoder, device, threshold)

    elif config['goal_achieving_criterion'] == 'state_similarity':
        ssim = torch.load(config['ssim_network_params.path'], map_location='cpu')
        device = torch.device(config['ssim_network_params.device'])
        threshold = config['ssim_network_params.threshold']
        return models.SSIMCriterion(ssim.ssim_network, device, threshold)
    else:
        raise AttributeError(f"unknown goal_achieving_criterion '{config['env.goal_achieving_criterion']}'")


def gen_navigation_env(conf, verbose=True, goal_achieving_criterion=None):
    env = minigrid_envs.gen_wrapped_env(conf, verbose=verbose)

    if not goal_achieving_criterion:
        goal_achieving_criterion = get_goal_achieving_criterion(conf)

    if conf.get('state_distance_network_params', {}).get('path', False):
        encoder = torch.load(conf['state_distance_network_params.path'], map_location='cpu')
        device = torch.device(conf['state_distance_network_params.device'])
        embedder = models.StateEmbedder(encoder, device)
    else:
        embedder = None

    if conf.get('goal_type', None) == 'random':
        random_goal_generator = minigrid_envs.random_grid_goal_generator(conf, verbose=verbose)
    else:
        random_goal_generator = None

    env = environments.navigation_wrapper(
        env=env,
        conf=conf,
        goal_achieving_criterion=goal_achieving_criterion,
        random_goal_generator=random_goal_generator,
        verbose=verbose)

    if conf.get('intrinsic_episodic_reward', False):
        env = environments.IntrinsicEpisodicReward(env, embedder)

    return env


def get_encoders(env, conf):
    grid_size = env.observation_space['image'].shape[0]
    state_encoder = encoders.get_encoder(grid_size, conf['worker'])
    goal_state_encoder = encoders.get_encoder(grid_size, conf['master'])
    return state_encoder, goal_state_encoder


####

import torch
from torch import nn

class GaussDropout(nn.Module):
    def __init__(self, alpha=0.983):
        super().__init__()
        self.alpha = alpha

    def forward(self, x):
        #noise = torch.randn_like(x)
        #x = x * (1 + self.alpha * noise)
        return x
####


def get_hindsight_state_encoder(state_encoder, goal_state_encoder, config):
    hidden_size_master = config['master']['head.hidden_size']
    emb_size = config['master']['emb_size']
    # FIXME: DDPG's actor looks strange here
    #goal_state_encoder = models.ActorNetwork(emb_size, goal_state_encoder, hidden_size_master)

    vae_model = VAE((64, 64, 3))

    goal_state_encoder = nn.Sequential(VAEEncoder(vae_model), GaussDropout())
    goal_state_encoder.output_size = 256
    return models.GoalStateEncoder(state_encoder=state_encoder, goal_state_encoder=goal_state_encoder)


def get_ppo_worker_agent(env, config):
    hindsight_encoder = get_hindsight_state_encoder(*get_encoders(env, config), config)
    hidden_size = config['worker.head.hidden_size']
    policy = models.ActorCriticNetwork(env.action_space, hindsight_encoder, hindsight_encoder, hidden_size, hidden_size)

    ppo = PPO(
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

    vae_state_dict = torch.load('vae.pt', map_location='cpu')
    ppo.actor_critic.actor.state_encoder.goal_state_encoder.get_submodule('0').vae.load_state_dict(vae_state_dict)
    ppo.actor_critic.critic.state_encoder.goal_state_encoder.get_submodule('0').vae.load_state_dict(vae_state_dict)

    return ppo


def get_rnd_model(config):
    if config['env'].get('fully_observed', True):
        grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)
    else:
        grid_size = 7 * config['env'].get('tile_size', 1)
    rnd = models.RNDModel(
        encoders.get_encoder(grid_size, config['rnd']),
        encoders.get_encoder(grid_size, config['rnd']),
        config['agent.device'])

    return rnd


def get_imppo_worker_agent(env, config):
    hindsight_encoder = get_hindsight_state_encoder(*get_encoders(config), config)
    hidden_size = config['worker.head.hidden_size']
    policy = models.ActorCriticNetwork(
        env.action_space, hindsight_encoder,
        hindsight_encoder, hidden_size,
        hidden_size, use_intrinsic_motivation=True
    )

    rnd = get_rnd_model(config)

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
        agent.to(config['agent.device'])

        logger.info(f"Running PPO agent training: {config['training.n_steps'] * config['training.n_processes']} steps")
        train_ppo(
            env=env,
            agent=agent,
            conf=config,
        )

    elif config['training.algorithm'] == 'imppo':
        agent = get_imppo_worker_agent(env, config)
        agent.to(config['agent.device'])

        steps = config['training.n_steps'] * config['training.n_processes']
        logger.info(f"Running {config['training.algorithm']} agent training: {steps} steps")
        im_train_ppo(
            env=env,
            agent=agent,
            conf=config,
        )

    else:
        raise AttributeError(f"unknown algorithm '{config['training.algorithm']}'")


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('rllr.env.wrappers')
    main()
