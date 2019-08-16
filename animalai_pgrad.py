import torch
from animalai.envs import ArenaConfig
from animalai.envs.gym.environment import AnimalAIEnv

import models
import pgrad
import train
import policy

env = AnimalAIEnv(
    environment_filename='env/AnimalAI',
    worker_id=1,
    n_arenas=1,
    arenas_configurations=ArenaConfig('animalai-conf/1-Food.yaml'),
    docker_training=False,
    retro=True)

nn = models.ConvPolicy(env)
optimizer = torch.optim.Adam(nn.parameters(), lr=0.01)
updater = pgrad.PGUpdater(optimizer, gamma=.99)
policy = policy.NNPolicy(nn, updater)

train.train_loop(env=env, policy=policy, n_episodes=1000)
