import gym
import torch

from navigation_models import ConvNavPolicy
from pgrad import PGUpdater
from policy import RandomActionPolicy, NNPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories, NavigationTrainer, ssim_dist, ssim_dist_euclidian
from constants import *

from gym_minigrid.wrappers import *
from custom_envs import *
from custom_wrappers import *

import matplotlib.pyplot as plt

env = gym.make('MiniGrid-MyEmpty-8x8-v0')
env = RGBImgAndStateObsWrapper(env)

explore_policy = RandomActionPolicy(env)

te = TrajectoryExplorer(env, explore_policy, n_steps=150, n_episodes=50)
tasks = generate_train_trajectories(te, n_initial_points=3, take_prob=.5)

model = torch.load(f'./saved_models/pretrained_navigation_model_{env.spec.id}.pkl')
model.to(DEVICE)
model.eval()
policy = NNPolicy(model, None)

def plt_show(states, names):
    fig=plt.figure(figsize=(4, 4))
    #fig=plt.figure()
    for i, state in enumerate(states):
        fig.add_subplot(1, len(states), i+1)
        plt.imshow(state['image'])
        plt.title(names[i])

    plt.show()
    input('\npress any key to start new task ...\n')
    plt.close()

with torch.no_grad():
    n_trivial_tasks, n_completed_tasks = 0, 0
    for i, (initial_trajectory, known_trajectory, desired_state) in enumerate(tasks):
        print('\n--------------------------------------------')
        print(f'NEW TASK {i}/{len(tasks)}:')

        state = env.reset()
        initial_state = state

        if initial_trajectory is not None:
            for a in initial_trajectory:
                print('    initial tragectory step')
                env.render()
                state, reward, done, _ = env.step(a)

        for j in range(32):

            if isinstance(state, np.ndarray):
                action, context = policy((state, desired_state))

            elif isinstance(state, dict):
                action, context = policy((state['image'], desired_state['image']))

            else:
                raise AttributeError(f'unsupported state type: {type(state)}')

            env.render()

            state, _, done, _ = env.step(action)

            sim =  ssim_dist_euclidian(state, desired_state)
            if sim > 0.999:
                if j == 0:
                    print('    trivial task')
                    n_trivial_tasks += 1
                else:
                    n_completed_tasks += 1
                print('    Completed')
                break
            elif done:
                print('    env Done')
                break

        print(f'    ACCURACY: {float(n_completed_tasks)/ (len(tasks)-n_trivial_tasks)}, n_trivial_tasks={n_trivial_tasks}, n_steps={j+1}')
        #plt_show([initial_state, state, desired_state], ['initial state', 'state', 'desired state'])
            

