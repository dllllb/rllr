import logging
import torch

from rllr.algo import IMPPO
from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper, ZeroRewardWrapper, HashCounterWrapper
from rllr.models import encoders, ActorCriticNetwork, StateSimilarityNetwork, RNDModel
from rllr.utils import im_train_ppo, get_conf, switch_reproducibility_on
from rllr.utils.state_similarity import ContrastiveStateSimilarity
from rllr.utils.logger import init_logger

logger = logging.getLogger(__name__)


def get_agent(env, config):
    grid_size = env.observation_space.shape[0]
    encoder = encoders.get_encoder(grid_size, config['worker'])
    hidden_size = config['worker.head.hidden_size']
    policy = ActorCriticNetwork(
        env.action_space, encoder,
        encoder, hidden_size,
        hidden_size, use_intrinsic_motivation=True,
        is_recurrent=True if config['worker.state_encoder_type'] == 'cnn_rnn' else False,
        same_encoders=True
    )

    rnd = RNDModel(
        encoders.get_encoder(grid_size, config['rnd']),
        encoders.get_encoder(grid_size, config['rnd']),
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


def get_ssim(env, conf):
    grid_size = env.observation_space.shape[0]
    encoder = encoders.get_encoder(grid_size, conf['state_similarity'])
    ssim_network = StateSimilarityNetwork(encoder, conf['state_similarity.hidden_size'])
    ssim = ContrastiveStateSimilarity(
        ssim_network,
        lr=conf['state_similarity.lr'],
        radius=conf['state_similarity.radius'],
        epochs=conf['state_similarity.epochs'],
    ).to(conf['agent.device'])
    return ssim


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed
    env = minigrid_envs.gen_wrapped_env(conf['env'], verbose=False)

    env = ZeroRewardWrapper(env)
    hash_penalty = conf.get('env.hash.hash_penalty', None)
    if hash_penalty is not None:
        rnd = None
        if conf['env.hash.hash_type'] == 'rnd':
            rnd_grid_size = env.observation_space.shape[0] // conf.get('env.hash.hash_pool_size', 12)
            rnd = encoders.get_encoder(rnd_grid_size, conf['env.hash.rnd'])
            rnd.to(conf.get('env.hash.device', 'cpu'))
        env = HashCounterWrapper(env, hash_penalty, conf['env.hash'], rnd=rnd)
    env = EpisodeInfoWrapper(env)

    return env


def train_ssim_with_rnd(env, agent, ssim, conf):
    def ssim_update_callback(rollouts):
        torch.save(ssim, conf['outputs.ssim_model'])
        return ssim.update(rollouts)

    im_train_ppo(env, agent, conf, ssim_update_callback)
    return agent, ssim


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
    ssim = get_ssim(env, config)

    logger.info(f"Running agent training: { config['training.n_steps'] * config['training.n_processes']} steps")
    train_ssim_with_rnd(
        env=env,
        agent=agent,
        ssim=ssim,
        conf=config
    )


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('gym_minigrid_navigation.environments')
    main()
