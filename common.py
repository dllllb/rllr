import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions import Categorical

class MLPPolicy(nn.Module):
    def __init__(self, env):
        super(MLPPolicy, self).__init__()
        self.state_space = env.observation_space.shape[0]
        self.action_space = env.action_space.n
        
        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)
        
        self.model = nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
        )
        
    def forward(self, x):    
        return self.model(x)

class PGUpdater:
    def __init__(self, optimizer, gamma):
        self.optimizer = optimizer
        self.gamma = gamma
        
    def __call__(self, pred_hist, reward_hist):
        # See https://github.com/pytorch/examples reinforcement_learning/reinforce.py
    
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in reward_hist[::-1]:
            R = r + self.gamma * R
            rewards.insert(0,R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        loss = []
        for log_prob, reward in zip(pred_hist, rewards):
            loss.append(-log_prob * reward)
        loss = torch.cat(loss).sum()

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    

def train_loop(env, policy, updater, n_eposodes):
    running_reward = 10
    for episode in range(n_eposodes):
        pred_hist, reward_hist, time, _ = play_episode(env, policy, render=False)
        
        updater(pred_hist, reward_hist)
        
        # Used to determine when the environment is solved
        running_reward = (running_reward * 0.99) + (time * 0.01)
        
        if episode % 50 == 0:
            print(f'Episode {episode}\tLast length: {time:5d}\tAverage length: {running_reward:.2f}')
        if running_reward > env.spec.reward_threshold:
            print(f"Solved! Running reward is now {running_reward} and the last episode runs to {time} time steps!")
            break
            

def play_episode(env, policy, render):
    state = env.reset() # Reset environment and record the starting state
    frames = []
    
    pred_hist = []
    reward_hist = []
    
    for time in range(1000):
        if render:
            frames.append(env.render(mode='rgb_array'))
        
        #Select an action by running policy model and choosing based on the probabilities in state
        state = Variable(torch.from_numpy(state).type(torch.FloatTensor))
        c = Categorical(policy(state))
        action = c.sample()
        
        # Add log probability of our chosen action to our history
        pred_hist.append(c.log_prob(action))

        # Step through environment using chosen action
        state, reward, done, _ = env.step(action.data[0])
        # Save reward
        reward_hist.append(reward)

        if done:
            break
            
    pred_hist = torch.cat(pred_hist)
    
    return pred_hist, reward_hist, time, frames
