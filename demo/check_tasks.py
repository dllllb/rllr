import os
import gym
from custom_envs import *
from custom_wrappers import *
import matplotlib.pyplot as plt
import pickle

def plt_show(states, names):
    fig=plt.figure(figsize=(4, 4))
    for i, state in enumerate(states):
        fig.add_subplot(1, len(states), i+1)
        plt.imshow(state['image'])
        plt.title(names[i])

    plt.show()
    #input('\npress any key to start new task ...\n')
    plt.close()

if not os.path.exists('./tasks.pkl'):
    raise FileNotFoundError('you do not have any saved tasks')

with open('./tasks.pkl', 'rb') as rfs:
    tasks = pickle.load(rfs)

env = gym.make('MiniGrid-MyEmptyRandomPosMetaAction-8x8-v0')
env = RGBImgAndStateObsWrapper(env)

for i, (initial_trajectory, known_trajectory, desired_state) in enumerate(tasks):
    start_state = env.reset()
    if initial_trajectory is not None:
        for a in initial_trajectory:
            state, reward, done, _ = env.step(a)

    plt_show([start_state, desired_state], ['start state', 'desired state'])