from rllr.algo import IMPPO
from rllr.env import EpisodeInfoWrapper, make_vec_envs, minigrid_envs
from rllr.models import encoders, ActorCriticNetwork, RNDModel
from rllr.utils import im_train_ppo
from rllr.utils import switch_reproducibility_on, get_conf
import gym
import numpy as np


class MissionWrapper(gym.Wrapper):
    def __init__(self, env, sent_len, vocab):
        super(MissionWrapper, self).__init__(env)
        self.observation_space = gym.spaces.Dict({
            'state': env.observation_space['image'],
            'mission': gym.spaces.Box(0, 1, (sent_len, len(vocab))),
        })

        self.words = np.array(vocab)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self.observation(obs)
        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.observation(self.env.reset())

    def observation(self, observation):
        mission = \
            (np.array([observation['mission'].split()]).reshape(-1, 1) == self.words).astype(np.float32)
        return {'state': observation['image'], 'mission': mission}


def gen_env_with_seed(conf, seed):
    env = gym.make(conf['env.env_task'])
    env.seed(seed)
    vocab = [
        'blue', 'purple', 'near', 'green', 'key', 'grey', 'ball', 'red', 'the', 'yellow', 'put', 'box'
    ]

    env = minigrid_envs.RGBImgObsWrapper(env, tile_size=conf['env.tile_size'])
    env = MissionWrapper(env, sent_len=8, vocab=vocab)
    return EpisodeInfoWrapper(env)


def get_agent(env, config):
    state_conf = config['encoder']
    hidden_size = state_conf['head']['hidden_size']
    state_encoder = encoders.ImageTextEncoder(env.observation_space['state'].shape, env.observation_space['mission'].shape)
    policy = ActorCriticNetwork(
        env.action_space, state_encoder,
        state_encoder, hidden_size,
        hidden_size, use_intrinsic_motivation=True
    )

    grid_size = env.observation_space['state'].shape[0]

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

    im_train_ppo(
        env=env,
        agent=agent,
        conf=config,
    )


if __name__ == '__main__':
    main()
