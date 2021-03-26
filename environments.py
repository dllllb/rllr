import gym
import logging
import numpy as np

from collections import deque
from gym.wrappers import Monitor
from scipy.stats import norm

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
            self.is_goal_achieved = self.goal_achieving_criterion(state, self.goal_state)
        else:
            self.is_goal_achieved = False
        return self.is_goal_achieved

    def step(self, action):
        next_state, _, _, info = self.env.step(action)
        done = self._goal_achieved(next_state) or (self.step_count >= self.max_steps)
        reward = 1 if self.is_goal_achieved else -0.1

        return next_state, reward, done, info


class RandomGoalWrapper(NavigationGoalWrapper):
    """
    Wrapper for setting random goal
    """
    def __init__(self, env, goal_achieving_criterion, random_goal_generator, verbose=False):
        self.verbose = verbose
        self.random_goal_generator = random_goal_generator
        super().__init__(env, goal_achieving_criterion)

    def reset(self):
        self.goal_state = next(self.random_goal_generator)
        self.is_goal_achieved = False
        return super().reset()


class FromBufferGoalWrapper(NavigationGoalWrapper):
    """
    Wrapper for setting goal from buffer
    """
    def __init__(self, env, goal_achieving_criterion, conf, verbose=False):
        self.buffer_size = conf['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)
        self.complexity_buffer = deque(maxlen=self.buffer_size)
        self.verbose = verbose
        self.complexity = conf['init_complexity']
        self.complexity_step = conf['complexity_step']
        self.threshold = conf['threshold']
        self.update_period = conf['update_period']
        self.max_complexity = conf['max_complexity']
        self.scale = conf['scale']
        self.episode_count = 0
        self.achieved_count = 0
        super().__init__(env, goal_achieving_criterion)

    def reset_buffer(self):
        self.buffer = deque(maxlen=self.buffer_size)

    def buffer_random_choice(self):
        steps_array = np.array([x for x, _ in self.buffer])
        p = norm.pdf(steps_array, loc=self.complexity, scale=self.scale)
        p /= p.sum()
        choice = np.random.choice(np.arange(len(steps_array)), p=p)
        _, goal_state = self.buffer[choice]

        return goal_state

    def reset(self):
        if self.episode_count % self.update_period == 0:
            self.update_complexity()
        self.episode_count += 1

        if len(self.buffer) < self.buffer_size:  # with out goal, only by max_steps episode completion
            return super().reset()

        else:
            state = super().reset()
            while True:
                # loop enforces goal_state != current state
                self.goal_state = self.buffer_random_choice()
                if not self._goal_achieved(state):
                    break

            if self.verbose:
                logger.info(f"Buffer goal: {self.goal_state['position'], self.goal_state['direction']}")

            return state

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        if self.is_goal_achieved:
            self.achieved_count += 1

        if self.goal_state is None:
            self.buffer.append((self.step_count, next_state))

        return next_state, reward, done, info

    def update_complexity(self):
        avg_achieved_goals = self.achieved_count / self.update_period
        if avg_achieved_goals >= self.threshold and self.complexity + self.complexity_step <= self.max_complexity:
            self.complexity += self.complexity_step
            logger.info(f"avg achieved goals: {avg_achieved_goals}, new complexity: {self.complexity}")
        self.achieved_count = 0


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
        env = FromBufferGoalWrapper(env, goal_achieving_criterion, conf['from_buffer_choice_params'], verbose=verbose)
    else:
        raise AttributeError(f"unknown goal_type '{goal_type}'")
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
