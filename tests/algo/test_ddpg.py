from rllr.algo.ddpg import DDPG
from rllr.models.encoders import MLP
from rllr.buffer.replay import ReplayBuffer
from rllr.models.models import ActorCriticNetwork
import torch
import gym


def get_ddpg_agent(env, device):
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    action_range = (env.action_space.low[0], env.action_space.high[0])
    replay_buffer = ReplayBuffer(buffer_size=int(2e6), batch_size=256, device=device)
    actor_state_encoder = MLP(input_size=state_size, hidden_layers_sizes=[128])
    critic_state_encoder = MLP(input_size=state_size, hidden_layers_sizes=[128])
    actor_critic = ActorCriticNetwork(action_size, actor_state_encoder, critic_state_encoder,
                                      actor_hidden_size=[128, 128], critic_hidden_size=[128, 128],
                                      action_range=action_range)
    return DDPG(actor_critic, replay_buffer, device, learning_rate_actor=5e-4, learning_rate_critic=5e-4)


def test_ddpg_mountain_car_continuous():
    device = torch.device("cpu")
    env = gym.make('MountainCarContinuous-v0')
    ddpg = get_ddpg_agent(env, device)
    assert ddpg.train(env, n_steps=40000, verbose=4)


def test_ddpg_pendulum():
    device = torch.device("cpu")
    env = gym.make('Pendulum-v0')
    ddpg = get_ddpg_agent(env, device)
    assert ddpg.train(env, n_steps=40000, verbose=4)