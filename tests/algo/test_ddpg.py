from rllr.algo.ddpg import DDPG
from rllr.models.ddpg import ActorCriticNetwork
from rllr.models.encoders import MLP
import torch
import gym
import numpy as np
import pytest


def get_ddpg_agent(env, device):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_range = (env.action_space.low[0], env.action_space.high[0])
    actor_state_encoder = MLP(input_size=state_size, hidden_layers_sizes=[128])
    critic_state_encoder = MLP(input_size=state_size, hidden_layers_sizes=[128])
    actor_critic = ActorCriticNetwork(action_size, actor_state_encoder, critic_state_encoder,
                                      actor_hidden_size=[128], critic_hidden_size=[128])
    return DDPG(actor_critic, device, buffer_size=1000000, batch_size=128, action_range=action_range,
                learning_rate_critic=1e-4, learning_rate_actor=1e-3, update_step=50, epochs=50)


#@pytest.mark.skip(reason="requires long running time, enable if needed")
def test_ddpg_mountain_car_continuous():
    device = torch.device("cpu")
    env = gym.make('MountainCarContinuous-v0')
    ddpg = get_ddpg_agent(env, device)
    assert ddpg.train(env, n_steps=40000, verbose=4)


#@pytest.mark.skip(reason="requires long running time, enable if needed")
def test_ddpg_pendulum():
    device = torch.device("cpu")
    env = gym.make('Pendulum-v0')
    ddpg = get_ddpg_agent(env, device)
    assert ddpg.train(env, n_steps=40000, verbose=4)


def test_scale_and_unscale():
    device = torch.device("cpu")
    env = gym.make('Pendulum-v0')
    ddpg = get_ddpg_agent(env, device)
    action = np.array([-2., 0., 2.])
    scaled_action = ddpg.scale_action(action)
    assert np.array_equal(scaled_action, np.array([-1., 0., 1.]))
    unscaled_action = ddpg.unscale_action(scaled_action)
    assert np.array_equal(unscaled_action, np.array([-2., 0. , 2.]))

