import gym
import torch
import cv2

from policy import RandomActionPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories


def visulize_trajectory(env, trajectory):
    env.reset()
    states = []

    initial_trajectory, known_trajectory, desired_state = trajectory

    t = initial_trajectory + known_trajectory

    for a in t:
        state, reward, done, _ = env.step(a)
        states.append(state)

    return states


env = gym.make('BreakoutDeterministic-v4')
env.seed(1)
torch.manual_seed(1)

explore_policy = RandomActionPolicy(env)

te = TrajectoryExplorer(env, explore_policy, n_episodes=500, n_steps=3)
tasks = generate_train_trajectories(te, 1, .5)

frames = visulize_trajectory(env, tasks[0])

for frame in frames:
    cv2.imshow('Trajectory', frame)
    cv2.waitKey(0)

cv2.destroyAllWindows()
