from rllr.algo.ppo import PPO
from rllr.models.ppo import ActorCritic
from rllr.models.encoders import MLP
import torch
import gym
import pytest


def get_ppo_agent(env, device):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    actor_state_encoder = MLP(input_size=state_size, hidden_layers_sizes=[128])
    critic_state_encoder = MLP(input_size=state_size, hidden_layers_sizes=[128])
    actor_critic = ActorCritic(action_size, actor_state_encoder, critic_state_encoder,
                               actor_hidden_size=[128], critic_hidden_size=[128])
    return PPO(actor_critic, device,
               memory_size=128,
               epochs=10,
               lr=3e-5,
               lamb=0.95,
               gamma=0.99,
               eps=0.2,
               c1=0.5,
               c2=0.01)

@pytest.mark.skip(reason="requires long running time, enable if needed")
def test_ppo_mountain_car():
    device = torch.device("cpu")
    env = gym.make('MountainCar-v0')
    ppo = get_ppo_agent(env, device)
    assert ppo.train(env, n_steps=100000, verbose=10)


@pytest.mark.skip(reason="requires long running time, enable if needed")
def test_ppo_cart_pole():
    device = torch.device("cpu")
    env = gym.make('CartPole-v0')
    ppo = get_ppo_agent(env, device)
    assert ppo.train(env, n_steps=10000, verbose=10)

