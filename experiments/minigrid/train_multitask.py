import logging

from rllr.utils import im_train_ppo
from rllr.env.vec_wrappers import make_vec_envs
from rllr.utils import switch_reproducibility_on, get_conf
from rllr.models import encoders
from rllr.models.ppo import ActorCriticNetwork
from rllr.algo import IMPPO
from rllr.models import RNDModel
from rllr.utils.logger import init_logger
import gym
from gym_minigrid.wrappers import RGBImgPartialObsWrapper
from rllr.env.gym_minigrid_navigation.environments import ImageObsWrapper
import numpy as np
from rllr.models.encoders import RNNEncoder, LastActionEncoder
from gym_minigrid.minigrid import MiniGridEnv

logger = logging.getLogger(__name__)


class MultitaskMinigridEnv:
    def __init__(self, tasks, tile_size, actions=('left', 'right', 'forward', 'toggle')):
        self.tasks = [
            ImageObsWrapper(RGBImgPartialObsWrapper(gym.make(env_name), tile_size=tile_size)) for env_name in tasks
        ]

        action_names = {v.name: int(v) for v in MiniGridEnv.Actions}
        self.action_ids = [action_names[name] for name in actions]

        self.action_space = gym.spaces.Discrete(len(actions))
        self.observation_space = gym.spaces.Dict({
            'state': self.tasks[0].observation_space,
            'last_action': gym.spaces.Discrete(self.action_space.n)
        })

        self.metadata = self.tasks[0].metadata

        self.reward_range = [np.inf, -np.inf]
        for t in self.tasks:
            self.reward_range[0] = min(self.reward_range[0], t.reward_range[0])
            self.reward_range[1] = max(self.reward_range[1], t.reward_range[1])

        self.task_names = {i: task for i, task in enumerate(tasks)}

        self.env_id = None
        self.env = None
        self.episode_reward = None
        self.episode_steps = None

    def step(self, action):
        action = int(action)
        action_id = self.action_ids[action]
        obs, reward, done, info = self.env.step(action_id)
        self.episode_reward += reward
        self.episode_steps += 1
        if done:
            info['episode'] = {
                'reward': self.episode_reward,
                'steps': self.episode_steps,
                'task': self.task_names[self.env_id]
            }
        return {'state': obs, 'last_action': action}, reward, done, info

    def reset(self, **kwargs):
        self.env_id = np.random.randint(len(self.tasks))
        self.env = self.tasks[self.env_id]
        self.episode_reward = 0
        self.episode_steps = 0
        return {'state': self.env.reset(**kwargs), 'last_action': 0}

    def seed(self, value=None):
        for t in self.tasks:
            t.seed(value)

    def render(self, mode='human'):
        return self.env.render(mode)


def gen_env_with_seed(conf, seed):
    env = MultitaskMinigridEnv(conf['env.tasks'], conf['env.tile_size'])
    env.action_space.np_random.seed(seed)
    env.seed(seed)

    return env


def get_agent(env, config):
    state_conf = config['encoder']
    hidden_size = state_conf['head']['hidden_size']
    grid_size = config['env.grid_size'] * config['env'].get('tile_size', 1)
    state_encoder = LastActionEncoder(RNNEncoder(
        encoders.get_encoder(
            grid_size,
            config['encoder']),
        state_conf['recurrent_hidden_size'] // 2
    ), env.action_space.n)
    policy = ActorCriticNetwork(
        env.action_space, state_encoder,
        state_encoder, hidden_size,
        hidden_size,
        use_intrinsic_motivation=True,
        is_recurrent=True
    )

    rnd = RNDModel(
        encoders.get_encoder(grid_size, config['rnd_encoder']),
        encoders.get_encoder(grid_size, config['rnd_encoder']),
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
