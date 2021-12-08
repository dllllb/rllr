import gym
import logging
import numpy as np
import torch

from collections import deque
from gym.wrappers import Monitor
from scipy.stats import norm
import torch.nn.functional as F
from collections import Counter

from ..buffer import ReplayBuffer
from ..exploration import EpisodicMemory
from ..utils import convert_to_torch

logger = logging.getLogger(__name__)


class NavigationGoalWrapper(gym.Wrapper):
    """
    Wrapper for navigation through environment to the goal
    """
    def __init__(self, env, goal_achieving_criterion):
        self.goal_state = None
        self.is_goal_achieved = False
        self.goal_achieving_criterion = goal_achieving_criterion
        super().__init__(env)

    def _goal_achieved(self, state):
        if self.goal_state is not None:
            return self.goal_achieving_criterion(state, self.goal_state)
        else:
            return False

    def gen_goal(self, state):
        while True:
            # loop enforces goal_state != current state
            self.gen_goal_(state)
            if not self._goal_achieved(state):
                break

    def gen_goal_(self, state):
        raise NotImplementedError

    def reset(self):
        state = super().reset()
        self.gen_goal(state)
        self.is_goal_achieved = False
        return state

    def step(self, action):
        next_state, reward, done, info = self.env.step(action)
        self.is_goal_achieved = self._goal_achieved(next_state)
        reward = int(self.is_goal_achieved)

        if self.is_goal_achieved:
            self.gen_goal(next_state)

        return next_state, reward, done, info


class RandomGoalWrapper(NavigationGoalWrapper):
    """
    Wrapper for setting random goal
    """
    def __init__(self, env, goal_achieving_criterion, random_goal_generator, verbose=False):
        self.verbose = verbose
        self.random_goal_generator = random_goal_generator
        super().__init__(env, goal_achieving_criterion)

    def gen_goal_(self, state):
        self.goal_state = next(self.random_goal_generator)


class FromBufferGoalWrapper(NavigationGoalWrapper):
    """
    Wrapper for setting goal from buffer
    """
    def __init__(self, env, goal_achieving_criterion, conf, verbose=False):
        self.buffer_size = conf['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)
        self.complexity_buffer = deque(maxlen=self.buffer_size)
        self.verbose = verbose
        self.warmup_steps = conf['warmup_steps']
        super().__init__(env, goal_achieving_criterion)

    def reset_buffer(self):
        self.buffer = deque(maxlen=self.buffer_size)

    def buffer_random_choice(self):
        choice = np.random.choice(np.arange(len(self.buffer)))
        goal_state = self.buffer[choice]

        return goal_state

    def gen_goal_(self, state):
        if len(self.buffer) >= self.warmup_steps:
            self.goal_state = self.buffer_random_choice()
        else:
            self.goal_state = None

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        self.buffer.append(next_state)

        return next_state, reward, done, info


class GoalObsWrapper(gym.core.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        if isinstance(env.observation_space, gym.spaces.dict.Dict):
            observation_space = self.observation_space.spaces['image']
        else:
            observation_space = env.observation_space
        self.observation_space = gym.spaces.Dict({
            'state': observation_space,
            'goal_state': observation_space,
        })

    def observation(self, obs):
        if isinstance(obs, dict):
            obs = obs['image']

        goal_obs = self.env.goal_state

        if goal_obs is None:
            goal_obs = obs

        if isinstance(goal_obs, dict):
            goal_obs = goal_obs['image']

        return {'state': obs, 'goal_state': goal_obs}


class SetRewardWrapper(gym.Wrapper):
    def __init__(self, env, reward_function):
        self.reward_function = reward_function
        self.pos_reward = hasattr(reward_function, 'is_pos_reward') and reward_function.is_pos_reward
        super().__init__(env)

    def step(self, action):
        state = self.env.observation(self.unwrapped.gen_obs())
        next_state, _, done, info = self.env.step(action)

        if self.pos_reward:
            reward = self.reward_function(state['position'], next_state['position'], self.goal_state['position'])
        else:
            reward = self.reward_function(state['image'], next_state['image'], self.goal_state['image'])

        return next_state, reward, done, info


def navigation_wrapper(env, conf, goal_achieving_criterion, random_goal_generator=None, verbose=False):
    goal_type = conf.get('goal_type', None)
    if goal_type == 'random':
        # env with random goal
        env = RandomGoalWrapper(
            env=env,
            goal_achieving_criterion=goal_achieving_criterion,
            random_goal_generator=random_goal_generator,
            verbose=verbose)

    elif goal_type == 'from_buffer':
        # env with goal from buffer
        env = FromBufferGoalWrapper(
            env,
            goal_achieving_criterion,
            conf['from_buffer_choice_params'],
            verbose=verbose)
    else:
        raise AttributeError(f"unknown goal_type '{goal_type}'")

    env = GoalObsWrapper(env)
    return env


class FullyRenderWrapper(gym.Wrapper):
    def render(self, *args, **kwargs):
        """This removes the default visualization of the partially observable field of view."""
        kwargs['highlight'] = False
        return self.unwrapped.render(*args, **kwargs)


def visualisation_wrapper(env, video_path):
    env = FullyRenderWrapper(env)  # removes the default visualization of the partially observable field of view.
    env = Monitor(env, video_path, force=True)
    return env


class IntrinsicEpisodicReward(gym.Wrapper):
    """
    Wrapper for adding Intrinsic Episode Memory Reward
        [Badia et al. (2020)](https://openreview.net/forum?id=Sye57xStvB)
    """
    def __init__(self, env, state_embedder=None, beta=0.3):
        self.beta = beta
        self.episodic_memory = EpisodicMemory()
        self.state_embedder = state_embedder if state_embedder is not None else lambda x: x
        super().__init__(env)

    def reset(self):
        state = super().reset()
        self.episodic_memory.clear()
        self.episodic_memory.add(self.state_embedder(state))
        return state

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        with torch.no_grad():
            embedded_state = self.state_embedder(state)
        intrinsic_reward = self.episodic_memory.compute_reward(embedded_state.unsqueeze(0)).item()
        self.episodic_memory.add(embedded_state)

        return state, reward + intrinsic_reward, done, info


class RandomNetworkDistillationReward(gym.Wrapper):
    """
    Wrapper for adding Random Network Distillation Reward
        [Burda et al. (2019)](https://arxiv.org/abs/1810.12894)
    """
    def __init__(self,
                 env,
                 target,
                 predictor,
                 device,
                 learning_rate=0.001,
                 buffer_size=10000,
                 batch_size=64,
                 update_step=4,
                 mean_gamma=0.99,
                 use_extrinsic_reward=False,
                 gamma=1):

        self.device = device
        self.target = target.to(device)
        self.predictor = predictor.to(device)
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size=buffer_size, batch_size=batch_size, device=device)
        self.steps_count = 0
        self.update_step = update_step
        self.running_mean = 0
        self.running_sd = 1
        self.mean_gamma = mean_gamma
        self.gamma = gamma
        self.alpha = 1
        self.eps = 1e-5
        self.use_extrinsic_reward = use_extrinsic_reward
        super().__init__(env)

    def reset(self):
        state = super().reset()
        return state

    def _learn(self):
        # Sample batch from buffer buffer
        states = list(self.replay_buffer.sample())[0]
        with torch.no_grad():
            targets = self.target(states)
        outputs = self.predictor(states)
        self.optimizer.zero_grad()
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()
        with torch.no_grad():
            intrinsic_reward = (targets - outputs).abs().pow(2).sum(1)
            self.running_mean = self.mean_gamma * self.running_mean + (1 - self.mean_gamma) * intrinsic_reward.mean()
            self.running_sd = self.mean_gamma * self.running_sd + (1 - self.mean_gamma) * intrinsic_reward.var().pow(0.5)
            self.alpha *= self.gamma

    def step(self, action):
        state, reward, done, info = self.env.step(action)

        if isinstance(state, dict):
            self.replay_buffer.add(state['state'], )
            converted_state = convert_to_torch([state['state']], device=self.device)
        else:
            self.replay_buffer.add(state, )
            converted_state = convert_to_torch([state], device=self.device)

        if self.alpha > self.eps:
            with torch.no_grad():
                intrinsic_reward = (self.target(converted_state) - self.predictor(converted_state)).abs().pow(2).sum()
                intrinsic_reward = (intrinsic_reward - self.running_mean) / self.running_sd
                intrinsic_reward = torch.clamp(intrinsic_reward, max=5, min=-5).item()

            self.steps_count = (self.steps_count + 1) % self.update_step
            if self.steps_count == 0:
                if self.replay_buffer.is_enough():
                    self._learn()
        else:
            intrinsic_reward = 0

        return state, reward * self.use_extrinsic_reward + self.alpha * intrinsic_reward, done, info


class EpisodeInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EpisodeInfoWrapper, self).__init__(env)
        self.episode_reward = 0
        self.episode_steps = 0
        self.visits_stats = dict()

    def reset(self):
        self.episode_reward = 0
        self.episode_steps = 0
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_steps += 1
        pos = tuple(self.agent_pos)
        if pos in self.visits_stats:
            self.visits_stats[pos] += 1
        else:
            self.visits_stats[pos] = 1
        info['visit_stats'] = self.visits_stats
        if done:
            info['episode'] = {'r': self.episode_reward, 'steps': self.episode_steps}
        return state, reward, done, info


class HierarchicalWrapper(gym.Wrapper):
    def __init__(self, env, low_level_policy, action_shape, n_steps=1):
        super(HierarchicalWrapper, self).__init__(env)
        self.policy = low_level_policy
        self.state = None
        self.n_steps = n_steps
        self.action_space = gym.spaces.Box(-1, 1, action_shape)

    def reset(self):
        self.state = self.env.reset()
        return torch.from_numpy(self.state)

    def step(self, action):
        cum_reward, step, done = 0, 0, False
        while not done and step < self.n_steps:
            _, low_action, _ = self.policy.act({
                'state': torch.from_numpy(self.state).unsqueeze(dim=0),
                'goal_emb': torch.from_numpy(action).unsqueeze(dim=0)
            }, deterministic=True)
            self.state, reward, done, info = self.env.step(low_action)
            cum_reward += reward
            step += 1
        return self.state, cum_reward, done, info
