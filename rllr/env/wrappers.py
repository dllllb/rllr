import gym
import logging
import numpy as np
import torch

from collections import deque, defaultdict
from gym.wrappers import Monitor
import torch.nn.functional as F
from rllr.models import encoders

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
        state = self.env.reset()
        self.gen_goal(state)
        self.is_goal_achieved = False
        return state

    def step(self, action):
        next_state, env_reward, done, info = self.env.step(action)
        self.is_goal_achieved = self._goal_achieved(next_state)
        reward = int(self.is_goal_achieved)

        #print("curr goal:", self.goal_state['position'][:2])

        if env_reward > 0 and done:
            done = False

        if self.is_goal_achieved and not done:
            self.gen_goal(next_state)

        return next_state, reward, done, info


class GoAndResetGoalGenerator(NavigationGoalWrapper):
    def __init__(self, env, goal_achieving_criterion, go_agent, conf, verbose=False):
        self.verbose = verbose
        super().__init__(env, goal_achieving_criterion)

        self.verbose = 200 if verbose else 0
        if self.verbose:
            self.count = np.random.randint(0, self.verbose)
        else:
            self.count = 0
        self.verbose_episode = False

        self.device = conf.get('device', 'cpu')
        self.go_agent = go_agent
        self.rhs_size = conf.get('rhs_size', 0)
        self.init_rhs = torch.zeros((1, self.rhs_size * 2), device=self.device)
        self.masks = torch.ones((1, 1), device=self.device)
        self.go_steps = conf['go_steps']
        self.go_deterministic = conf.get('go_deterministic', False)

    def go(self, next_obs):
        rnn_rhs = self.init_rhs
        for _ in range(self.go_steps):
            obs = next_obs
            _, action, _, rnn_hxs = self.go_agent.act(torch.tensor(next_obs['image']).unsqueeze(0),
                                                      rnn_rhs,
                                                      self.masks,
                                                      deterministic=self.go_deterministic)
            self.env.unwrapped.step_count -= 1
            next_obs, reward, done, info = self.env.step(action)
            if reward < 0:
                next_obs = obs
                done = False
                break
            if done and reward == 0:
                next_obs = None
                break
        return next_obs, done

    def gen_goal(self, state):
        while True:
            # loop enforces goal_state != current state
            done = self.gen_goal_(state)
            if not self._goal_achieved(state):
                break
        return done

    def gen_goal_(self, state):
        curr_pos = self.unwrapped.agent_pos
        curr_dir = self.unwrapped.agent_dir
        self.goal_state, done = self.go(state)
        self.unwrapped.agent_pos = curr_pos
        self.unwrapped.agent_dir = curr_dir
        return done

    def step(self, action):
        next_state, env_reward, done, info = self.env.step(action)
        self.is_goal_achieved = self._goal_achieved(next_state)
        reward = int(self.is_goal_achieved)

        if env_reward > 0 and done:
            done = False

        if self.is_goal_achieved and not done:
            done = self.gen_goal(next_state)

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
    def __init__(self, env, goal_achieving_criterion, conf, verbose=False, seed=0):
        self.buffer_size = conf['buffer_size']
        self.buffer = deque(maxlen=self.buffer_size)
        self.shrink_traj = conf.get("shrink_traj", 1)

        if 'unachieved_prob' in conf:
            self.unachieved_prob = conf['unachieved_prob']
            self.unachieved_buffer_size = conf['unachieved_buffer_size']
            self.unachieved_buffer = deque(maxlen=self.unachieved_buffer_size)

        self.warmup_steps = conf['warmup_steps']
        super().__init__(env, goal_achieving_criterion)

        self.count = 0
        self.flag = 0
        self.seed = seed
        np.random.seed(seed)

        self.verbose = 200 if verbose else 0
        if self.verbose:
            self.count = np.random.randint(0, self.verbose)
        else:
            self.count = 0
        self.verbose_episode = False

    def gen_goal(self, state):
        super().gen_goal(state)

        if self.verbose_episode:
            print(self.seed, self.goal_state['position'], self.flag)

    def reset(self):
        self.count += 1
        if self.verbose:
            self.verbose_episode = self.count % self.verbose == 0 and self.goal_state is not None

        if not self.is_goal_achieved and self.goal_state is not None:
            self.unachieved_buffer.append(self.goal_state)

        return super().reset()

    def reset_buffer(self):
        self.buffer = deque(maxlen=self.buffer_size)
        self.unachieved_buffer = deque(maxlen=self.unachieved_buffer_size)

    def buffer_random_choice(self):
        if self.unachieved_buffer and np.random.rand() < self.unachieved_prob:
            choice = np.random.choice(np.arange(len(self.unachieved_buffer)))
            goal_state = self.unachieved_buffer[choice]
            self.flag = 1
        else:
            choice = np.random.choice(np.arange(len(self.buffer)))
            goal_state = self.buffer[choice]
            self.flag = 0

        return goal_state

    def gen_goal_(self, state):
        if len(self.buffer) >= self.warmup_steps:
            self.goal_state = self.buffer_random_choice()
        else:
            self.goal_state = None

    def step(self, action):
        next_state, reward, done, info = super().step(action)
        if np.random.rand() < self.shrink_traj:
            self.buffer.append(next_state)

        return next_state, reward, done, info


class RNDBufferWrapper(NavigationGoalWrapper):

    def __init__(self, env, goal_achieving_criterion, conf, grid_size, device, verbose=1):

        super().__init__(env, goal_achieving_criterion)
        self.verbose = verbose
        self.device = device
        self.warmup_steps = conf['warmup_steps']
        goal_buffer_size = conf['goal_buffer_size']
        replay_buffer_size = conf['replay_buffer_size']
        self.goal_buffer = deque(maxlen=goal_buffer_size)
        learning_rate = conf['learning_rate']
        batch_size = conf['batch_size']
        update_step = conf['update_step']
        target_network = encoders.get_encoder(grid_size, conf['target'])
        predictor_network = encoders.get_encoder(grid_size, conf['predictor'])
        self.target = target_network.to(device)
        self.predictor = predictor_network.to(device)

        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_size=replay_buffer_size, batch_size=batch_size, device=device)
        self.steps_count = 0
        self.update_step = update_step
        self.explore_steps = conf.get('replay_buffer_size', 10)

    def _learn(self):
        states = list(self.replay_buffer.sample())[0]
        with torch.no_grad():
            targets = self.target(states)
        outputs = self.predictor(states)
        self.optimizer.zero_grad()
        loss = F.mse_loss(outputs, targets)
        loss.backward()
        self.optimizer.step()

    def gen_goal_(self, state, scores=None):
        if len(self.goal_buffer) >= self.warmup_steps:
            if scores is None:
                scores = self._get_scores()
            choice = np.random.choice(np.arange(len(self.goal_buffer)), p=scores)
            self.goal_state = self.goal_buffer[choice]
        else:
            self.goal_state = None

    def gen_goal(self, state):
        scores = self._get_scores()
        while True:
            # loop enforces goal_state != current state
            self.gen_goal_(state, scores)
            if not self._goal_achieved(state):
                break

    def _get_scores(self):
        if len(self.goal_buffer) == 0:
            return None
        states = convert_to_torch([state['image'] for state in self.goal_buffer])
        with torch.no_grad():
            rnd_scores = (self.target(states) - self.predictor(states)).abs().pow(2).sum(1)
        scores = rnd_scores.numpy()
        return scores/sum(scores)

    def step(self, action):
        state, reward, done, info = super().step(action)
        self.goal_buffer.append(state)
        self.replay_buffer.add(state['image'], )
        self.steps_count += 1
        if (self.steps_count % self.update_step) == 0:
            self.steps_count = 0
            if self.replay_buffer.is_enough():
                self._learn()
        return state, reward, done, info

    def reset(self):
        return super().reset()


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
            verbose=verbose,
            seed=conf['seed'])

    elif goal_type == 'rnd_buffer':
        device = conf['device']
        if conf['env_type'] == 'gym_minigrid':
            grid_size = conf['grid_size'] * conf.get('tile_size', 1)
        else:
            raise AttributeError(f"unknown env_type '{conf['env_type']}'")
        env = RNDBufferWrapper(env,
                               goal_achieving_criterion,
                               conf['rnd_buffer_params'],
                               grid_size,
                               device,
                               verbose=verbose)

    elif goal_type == "go_and_reset":
        go_agent = torch.load(conf['go_and_reset_params.go_agent'],
                              map_location=conf.get('go_and_reset_params.device', 'cpu'))
        env = GoAndResetGoalGenerator(
            env=env,
            goal_achieving_criterion=goal_achieving_criterion,
            go_agent=go_agent,
            conf=conf['go_and_reset_params'],
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


class EpisodeInfoWrapper(gym.Wrapper):
    def __init__(self, env):
        super(EpisodeInfoWrapper, self).__init__(env)
        self.episode_reward = 0
        self.episode_steps = 0
        self.visits_stats = defaultdict(int)

    def reset(self):
        self.episode_reward = 0
        self.episode_steps = 0
        self.visits_stats.clear()
        return self.env.reset()

    def step(self, action):
        state, reward, done, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_steps += 1
        self.visits_stats[tuple(self.agent_pos)] += 1
        info['visit_stats'] = self.visits_stats
        if done:
            info['episode'] = {
                'task': self.env.unwrapped.spec.id,
                'r': self.episode_reward,
                'steps': self.episode_steps
            }
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
        cum_reward, step, done, info = 0, 0, False, {}
        while not done and step < self.n_steps:
            _, low_action, _, _ = self.policy.act({
                'state': torch.from_numpy(self.state).unsqueeze(dim=0),
                'goal_emb': torch.from_numpy(action).unsqueeze(dim=0)
            }, None, None, deterministic=True)
            self.state, reward, done, info = self.env.step(low_action)
            cum_reward += reward
            step += 1
        return self.state, cum_reward, done, info


class ZeroRewardWrapper(gym.Wrapper):
    def step(self, action):
        state, reward, done, info = self.env.step(action)
        if reward > 0 and done:
            done = False

        return state, 0, done, info


class HashCounterWrapper(gym.Wrapper):
    def __init__(self, env, penalty, conf):
        super(HashCounterWrapper, self).__init__(env)
        self.hashed_states = set()
        self.penalty = penalty
        self.hash_type = conf.get('hash_type', 'simple')
        self.pool_size = conf.get('hash_pool_size', 12)

        if self.hash_type == "rnd":
            self.rnd = encoders.get_encoder(conf['rnd.grid_size'], conf['rnd'])
            self.rnd.to(conf.get('rnd.device', 'cpu'))

    def reset(self):
        obs = self.env.reset()
        self.hashed_states = set()
        if self.hash_type == "coord":
            self.hashed_states.add(4)
        self.hashed_states.add(self.get_hash(obs))
        return obs

    def get_hash(self, obs):
        if self.hash_type == 'simple':
            state_h = self.env.unwrapped.hash()
        elif self.hash_type == 'go_explore':
            img = torch.tensor(obs).unsqueeze(0)
            img = torch.permute(img, (0, 3, 1, 2))
            img = torch.max_pool2d(img, self.pool_size).numpy()
            #print(img.shape)
            img.flags.writeable = False
            state_h = hash(img.tobytes())
        elif self.hash_type == "coord":
            x, y = self.env.unwrapped.agent_pos
            if x < 9 and y < 9:
                state_h = 0
            elif x > 9 and y < 9:
                state_h = 1
            elif x < 9 and y > 9:
                state_h = 2
            elif x > 9 and y > 9:
                state_h = 3
            else:
                state_h = 4
        elif self.hash_type == "rnd":
            img = torch.tensor(obs).unsqueeze(0)
            img = torch.permute(img, (0, 3, 1, 2))
            with torch.no_grad():
                state_h = self.rnd(img).cpu().item()
        return state_h

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        state_h = self.get_hash(obs)
        if state_h in self.hashed_states:
            if not (self.hash_type == "coord" and state_h == 4):
                #reward -= self.penalty/100
                pass
        else:
            self.hashed_states.add(state_h)
            if not (self.hash_type == "coord" and state_h == 4):
                reward += self.penalty
        return obs, reward, done, info
