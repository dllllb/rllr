import logging
import torch

from rllr.buffer.rollout import RolloutStorage
from rllr.env import make_vec_envs, minigrid_envs, EpisodeInfoWrapper
from rllr.models import encoders, StateSimilarityNetwork
from rllr.utils import  get_conf, switch_reproducibility_on
from rllr.utils.state_similarity import ContrastiveStateSimilarity
from rllr.utils.logger import init_logger
from tqdm import trange

logger = logging.getLogger(__name__)


def get_ssim(env, conf):
    grid_size = env.observation_space.shape[0]
    encoder = encoders.get_encoder(grid_size, conf['state_similarity'])
    ssim_network = StateSimilarityNetwork(encoder, conf['state_similarity.hidden_size'])
    ssim = ContrastiveStateSimilarity(
        ssim_network,
        lr=conf['state_similarity.lr'],
        radius=conf['state_similarity.radius'],
        epochs=conf['state_similarity.epochs'],
    )
    return ssim


def gen_env_with_seed(conf, seed):
    conf['env.deterministic'] = True
    conf['env']['seed'] = seed
    env = minigrid_envs.gen_wrapped_env(conf['env'], verbose=False)
    env = EpisodeInfoWrapper(env)
    return env


def main(args=None):
    init_logger(__name__)
    config = get_conf(args)
    switch_reproducibility_on(config['seed'])

    env = make_vec_envs(
        lambda env_id: lambda: gen_env_with_seed(config, env_id),
        config['training.n_processes'],
        config['agent.device']
    )

    conf = config
    ssim = get_ssim(env, config)

    # training starts
    rollouts = RolloutStorage(
        conf['training.n_steps'], conf['training.n_processes'], env.observation_space, env.action_space,
    )

    obs = env.reset()
    rollouts.set_first_obs(obs)
    rollouts.to(conf['agent.device'])

    num_updates = int(conf['training.n_env_steps'] // conf['training.n_steps'] // conf['training.n_processes'])

    for epoch in trange(num_updates):
        for step in range(conf['training.n_steps']):
            # Sample actions
            action = torch.randint(0, 3, (conf['training.n_processes'],)).view(-1, 1)
            obs, reward, done, infos = env.step(action)

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            rollouts.insert(
                obs,
                action,
                torch.zeros_like(action),
                torch.zeros_like(action),
                reward,
                masks
            )

        shape = rollouts.obs.shape
        obs = rollouts.obs.transpose(1, 0).reshape(shape[0] * shape[1], *shape[2:])
        dones = 1 - rollouts.masks.transpose(1, 0).reshape(-1)
        print('ssim_loss', ssim.update(obs, dones))
        rollouts.after_update()
        torch.save(ssim, conf['outputs.model'])


if __name__ == '__main__':
    init_logger(__name__)
    init_logger('gym_minigrid_navigation.environments')
    main()
