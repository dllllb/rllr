import logging
import torch
from torch import nn

from rllr.algo import PPOGAN
from rllr.env import make_vec_envs, HierarchicalWrapper, EpisodeInfoWrapper, IntrinsicEpisodicReward, minigrid_envs
from rllr.models import encoders, ActorCriticNetwork, StateEmbedder
from rllr.utils import train_ppo, get_conf, switch_reproducibility_on
from rllr.utils.logger import init_logger
from vae import VAE

from train_worker_vae import GaussDropout

logger = logging.getLogger(__name__)


def init_params(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    if isinstance(m, nn.LSTM):
        for name, param in m.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)
    if isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias, 0)


class VAEEncoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.vae.encode(x)
        return x


class VAEDecoder(nn.Module):
    def __init__(self, vae):
        super().__init__()
        self.vae = vae
        for p in self.vae.parameters():
            p.requires_grad = False

    def forward(self, x):
        with torch.no_grad():
            x = self.vae.decode(x)
        return x


def get_master_agent(env, conf):
    #vae_model = torch.load(conf['vae']['path'])
    vae_model = VAE((64, 64, 3))
    vae_state_dict = torch.load('vae.pt', map_location='cpu')
    vae_model.load_state_dict(vae_state_dict)

    encoder = VAEEncoder(vae_model)
    decoder = VAEDecoder(vae_model)

    encoder.vae.load_state_dict(vae_state_dict)
    decoder.vae.load_state_dict(vae_state_dict)

    grid_size = env.observation_space.shape[0]
    actor_state_encoder = nn.Sequential(encoder,
                                        encoders.get_encoder(grid_size, conf['master'], (256,)))
    actor_state_encoder.output_size = 512
    critic_state_encoder = nn.Sequential(encoder,
                                         encoders.get_encoder(grid_size, conf['master'], (256,)))
    critic_state_encoder.output_size = 512

    hidden_size = conf['master.head.hidden_size']
    master_network = ActorCriticNetwork(
        env.action_space, actor_state_encoder, critic_state_encoder,
        actor_hidden_size=hidden_size, critic_hidden_size=hidden_size,
        same_encoders=True
    )
    master_network.apply(init_params)

    discriminator = encoders.get_encoder(grid_size, conf['discriminator'], (256,))
    discriminator.apply(init_params)

    master_agent = PPOGAN(
        master_network,
        encoder,
        decoder,
        discriminator,
        conf['agent.clip_param'],
        conf['agent.ppo_epoch'],
        conf['agent.num_mini_batch'],
        conf['agent.value_loss_coef'],
        conf['agent.entropy_coef'],
        conf['agent.lr'],
        conf['agent.eps'],
        conf['agent.max_grad_norm']
    )

    master_agent.encoder.vae.load_state_dict(vae_state_dict)
    master_agent.decoder.vae.load_state_dict(vae_state_dict)
    master_agent.actor_critic.actor.state_encoder.get_submodule('0').vae.load_state_dict(vae_state_dict)
    master_agent.actor_critic.critic.state_encoder.get_submodule('0').vae.load_state_dict(vae_state_dict)
    return master_agent


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed

    worker_agent = torch.load(conf['worker_agent.path'], map_location='cpu').to(conf['agent.device'])
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
