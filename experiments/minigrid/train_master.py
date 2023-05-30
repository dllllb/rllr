import logging
from pathlib import Path
import torch

from rllr.algo import PPO
from rllr.env import make_vec_envs, HierarchicalWrapper, EpisodeInfoWrapper, IntrinsicEpisodicReward, minigrid_envs
from rllr.models import encoders, ActorCriticNetwork, StateEmbedder
from rllr.utils import train_ppo, get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger

logger = logging.getLogger(__name__)


def get_master_agent(env, conf):
    grid_size = env.observation_space.shape[0]
    state_encoder = encoders.get_encoder(grid_size, conf['master'])
    goal_state_encoder = encoders.get_encoder(grid_size, conf['master'])

    hidden_size = conf['master.head.hidden_size']
    master_network = ActorCriticNetwork(
        env.action_space, state_encoder, goal_state_encoder,
        actor_hidden_size=hidden_size, critic_hidden_size=hidden_size
    )

    master_agent = PPO(
        master_network,
        conf['agent.clip_param'],
        conf['agent.ppo_epoch'],
        conf['agent.num_mini_batch'],
        conf['agent.value_loss_coef'],
        conf['agent.entropy_coef'],
        conf['agent.lr'],
        conf['agent.eps'],
        conf['agent.max_grad_norm']
    )
    return master_agent


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed

    worker_agent = torch.load(Path(__file__).parent / conf['worker_agent.path'], map_location='cpu').to(conf['agent.device'])
    emb_size = worker_agent.actor_critic.actor.state_encoder.goal_state_encoder.output_size

    env = minigrid_envs.gen_wrapped_env(conf['env'], verbose=False)

    if conf['env'].get('intrinsic_episodic_reward', False):
        encoder = worker_agent.actor_critic.actor.state_encoder.state_encoder
        device = torch.device(conf['agent.device'])
        embedder = StateEmbedder(encoder, device)
        env = IntrinsicEpisodicReward(env, embedder, conf['env.intrinsic_episodic_reward.gamma'])

    env = EpisodeInfoWrapper(env)
    return HierarchicalWrapper(
        env, worker_agent, (emb_size,), n_steps=conf['training'].get('worker_n_steps', 1)
    )


def main(args=None):
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    master_agent = get_master_agent(env, config)
    master_agent.to(config['agent.device'])

    logger.info(f"Running agent training: { config['training.n_steps'] * config['training.n_processes']} steps")
    train_ppo(env, master_agent, config)


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('ppo')
    init_logger('rllr.env.wrappers')
    init_logger('rllr.env.gym_minigrid_navigation.environments')
    main()
