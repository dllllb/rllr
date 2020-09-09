import gym
import time

env = gym.make('BreakoutDeterministic-v4')
#env = gym.make('CartPole-v0')
for k in range(10):
    env.reset()

    env.render()
    time.sleep(0.1)
    print(f'------------------------- reset environment {k}----------------------------\n')

    total_reward = 0
    for t in range(1000):

        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward

        env.render()
        time.sleep(1.0/2)

        if done:
            print("Episode finished after {} timesteps with reward {}".format(t+1, total_reward))
            break
env.close()