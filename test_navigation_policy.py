import gym
import torch

from navigation_models import ConvNavPolicy, StateAPINavPolicy
from pgrad import PGUpdater
from policy import RandomActionPolicy, NNPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories, NavigationTrainer, ssim_dist, ssim_dist_euclidian, ssim_dist_abs
from constants import *

from gym_minigrid.wrappers import *
from custom_envs import *
from custom_wrappers import *
import time
import warnings
import pickle

import matplotlib.pyplot as plt

render_ = False
show_task = False

#env = gym.make('MiniGrid-MyEmpty-8x8-v0')
env = gym.make('MiniGrid-MyEmptyRandomPos-8x8-v0')
env = RGBImgAndStateObsWrapper(env)

explore_policy = RandomActionPolicy(env)

#te = TrajectoryExplorer(env, explore_policy, n_steps=150, n_episodes=50)
#tasks = generate_train_trajectories(te, n_initial_points=3, take_prob=.5)
with open('./tasks.pkl', 'rb') as rfs:
    tasks = pickle.load(rfs)
warnings.warn('using casched dataset for navigation policy task')

model = torch.load(f'./saved_models/pretrained_navigation_model_{env.spec.id}.pkl')
model.to(DEVICE)
model.eval()
policy = NNPolicy(model, None)

def plt_show(states, names):
    if not render_ or not show_task:
        return
    fig=plt.figure(figsize=(4, 4))
    #fig=plt.figure()
    for i, state in enumerate(states):
        fig.add_subplot(1, len(states), i+1)
        plt.imshow(state['image'])
        plt.title(names[i])

    plt.show()
    input('\npress any key to start new task ...\n')
    plt.close()

def render(env):
    if render_:
        env.render()

with torch.no_grad():
    n_trivial_tasks, n_completed_tasks = 0, 0
    for i, (initial_trajectory, known_trajectory, desired_state) in enumerate(tasks):
        print('\n.................................................................')
        print(f'NEW TASK {i}/{len(tasks)}:')

        state = env.reset()
        initial_state = state

        #plt_show([initial_state, desired_state], ['initial state', 'desired state'])

        if initial_trajectory is not None:
            for a in initial_trajectory:
                print('    initial tragectory step')
                render(env)
                state, reward, done, _ = env.step(a)

        best_sim =  ssim_dist_abs(state, desired_state)
        if best_sim == 0:#0.999:
            print('    trivial task')
            n_trivial_tasks += 1
            continue
        
        trajectory_way = ''
        for j in range(140):
            action, context = policy((state, desired_state))

            state, _, done, _ = env.step(action)

            render(env)
            #time.sleep(1/ 3.0)

            sim = ssim_dist_abs(state, desired_state)

            if sim < best_sim:
                trajectory_way += '+'
                best_sim = sim
            else:
                trajectory_way += '-'

            if sim == 0:# 0.999:
                n_completed_tasks += 1
                break
            elif done:
                #print('    env Done')
                break

        print(f'    {trajectory_way}')
        prefix = '' if best_sim != 0 else '***COMPLETED*** '
        print(f'    {prefix}ACCURACY: {float(n_completed_tasks)/ (len(tasks)-n_trivial_tasks)}, n_trivial_tasks={n_trivial_tasks}, n_steps={j+1}')
        plt_show([initial_state, state, desired_state], ['initial state', 'state', 'desired state'])
            

