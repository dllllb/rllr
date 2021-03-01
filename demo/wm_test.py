import gym
import torch

import pgrad
import wmpolicy
from policy import RandomActionPolicy
from statesearch import TrajectoryExplorer, generate_train_trajectories, NavigationTrainer, mse_dist


def test_train_wm():
    env = gym.make('CartPole-v1')
    env.seed(1)
    torch.manual_seed(1)

    explore_policy = RandomActionPolicy(env)

    embed_size = 32
    encoder_nn = wmpolicy.mlp_encoder(env, embed_size)
    nav_nn = pgrad.MLPPolicy(env)

    np_optimizer = torch.optim.Adam(nav_nn.parameters(), lr=0.01)
    np_updater = pgrad.PGUpdater(np_optimizer, gamma=.99)
    policy = pgrad.NNPolicy(nav_nn, np_updater)

    te = TrajectoryExplorer(env, explore_policy, 5, 2)
    tasks = generate_train_trajectories(te, 3, .5)

    nt = NavigationTrainer(env, policy, n_steps_per_episode=3, state_dist=mse_dist)
    nt(tasks)

    sp_nn = wmpolicy.MLPNextStatePred(encoder_nn, embed_size)
    sp_optimizer = torch.optim.Adam(list(sp_nn.parameters()) + list(encoder_nn.parameters()), lr=0.01)
    sp_updater = wmpolicy.DistUpdater(sp_optimizer)
    policy = wmpolicy.SPPolicy(sp_nn, nav_nn, encoder_nn, sp_updater)

    nt = NavigationTrainer(env, policy, n_steps_per_episode=3, state_dist=mse_dist)
    nt(tasks)

    rp_nn = wmpolicy.MLPRewardPred(encoder_nn, embed_size)
    mse_optimizer = torch.optim.Adam(list(rp_nn.parameters()) + list(encoder_nn.parameters()), lr=0.01)
    mse_updater = wmpolicy.MSEUpdater(mse_optimizer)
    policy = wmpolicy.RPPolicy(rp_nn, nav_nn, mse_updater)

    nt = NavigationTrainer(env, policy, n_steps_per_episode=3, state_dist=mse_dist)
    nt(tasks)
