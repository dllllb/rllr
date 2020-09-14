import gym
import time

from gym_minigrid.wrappers import *
from gym_minigrid.minigrid import *
from custom_envs import *
import cv2
from custom_wrappers import *

ACTIONS = {
    0: 'Turn left',
    1: 'Turn right',
    2: 'Move forward',
    3: 'Pick up an object',
    4: 'Drop the object being carried',
    5: 'Toggle',
    6: 'Done',
}


#--------------------------------------------------------------
#env = gym.make('MiniGrid-Empty-8x8-v0')
env = gym.make('MiniGrid-MyEmpty-8x8-v0')
env = RGBImgAndStateObsWrapper(env)
#env = RGBImgObsWrapper(env) # Get pixel observations
#env = ImgObsWrapper(env) # Get rid of the 'mission' field


#--------------------------------------------------------------
#env = gym.make('BreakoutDeterministic-v4')

#--------------------------------------------------------------
#env = gym.make('CartPole-v0')

for k in range(10):
    env.reset()

    #env.render()
    print(f'------------------------- reset environment {k}----------------------------\n')

    total_reward = 0
    for t in range(1000):

        action = env.action_space.sample()
        
        observation, reward, done, info = env.step(action)
        agent_pos = observation['agent_pos']
        total_reward += reward

        #cv2.imwrite(f'./state.jpg', observation['image'])
        #print(ACTIONS[action])
        env.render()
        #input('...')
        
        #time.sleep(5.0)

        if done:
            print("Episode finished after {} timesteps with reward {}".format(t+1, total_reward))
            break
env.close()