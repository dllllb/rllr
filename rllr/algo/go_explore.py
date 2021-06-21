from collections import deque
import numpy as np


class GoExploreArchive:

    def __init__(self, archive_size, goal_achieving_criterion):
        self.archive_size = archive_size
        self.states = deque(maxlen=self.archive_size)
        self.visits = deque(maxlen=self.archive_size)
        self.scores = deque(maxlen=self.archive_size)
        self.goal_achieving_criterion = goal_achieving_criterion

    def add(self, state, reward):
        found = False
        for i in range(len(self.states)):
            if self.goal_achieving_criterion(state, self.states[i]):
                self.visits[i] += 1
                self.scores[i] += reward
                found = True

        if not found:
            self.states.append(state)
            self.visits.append(1)
            self.scores.append(reward)
        assert len(self.visits) == len(self.states)

    def sample(self):
        s = np.array([s+1e-8 for s in self.scores])
        s /= s.sum()
        v = np.array([1./np.sqrt(v+1) for v in self.visits])
        v /= v.sum()
        p = (v + s)/2
        p /= p.sum()
        choice = np.random.choice(np.arange(len(self.visits)), p=p)
        sampled_state = self.states[choice]
        return sampled_state


class GoExplore():

    def __init__(self, worker, goal_achieving_criterion, archive_size=10000, explore_steps=50, go_steps=50):
        self.worker = worker
        self.explore_steps = explore_steps
        self.go_steps = go_steps
        self.state_archive = GoExploreArchive(archive_size, goal_achieving_criterion)
        self.goal_achieving_criterion = goal_achieving_criterion

    def _get_to_goal(self, env, state, goal_state):

        steps = 0
        go_reward = 0

        for i in range(self.go_steps):
            action = self.worker.act({'state': state['image'], 'goal_state': goal_state['image']})
            next_state, reward, done, _ = env.step(action)
            state = next_state
            self.state_archive.add(state, reward)
            go_reward += reward
            steps += 1
            if done or self.goal_achieving_criterion(state, goal_state):
                break
        return go_reward, done, steps

    def go(self, env, state):
        # Buffer is empty first need some exploration
        if len(self.state_archive.states) == 0:
            go_reward, done, steps, state = self.explore(env)
            if done:
                return go_reward, done, steps

        # Sample goal state from buffer
        goal_state = self.state_archive.sample()

        # Go using worker to the state and update buffer
        go_reward, done, steps = self._get_to_goal(env, state, goal_state)
        return go_reward, done, steps

    def explore(self, env):
        # Make explore_steps with random action
        explore_reward = 0
        steps = 0
        for _ in range(self.explore_steps):
            action = env.action_space.sample()
            state, reward, done, _ = env.step(action)
            self.state_archive.add(state, reward)
            explore_reward += reward
            steps += 1
            if done:
                break
        return explore_reward, done, steps, state