import gym
import nle
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PosObsWrapper(gym.core.ObservationWrapper):
    """
    Add agent position to state dict
    """
    def observation(self, observation):
        observation['position'] = np.array([observation['blstats'][1], observation['blstats'][0]])
        return observation


def gen_wrapped_env(conf, verbose=False):
    env = gym.make('NetHackScore-v0')

    if conf.get('goal_achieving_criterion', None) == 'position' or verbose:
        env = PosObsWrapper(env)

    return env
