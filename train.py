import torch
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events

from constants import *
import time


def train_loop(env, policy, n_episodes, episode_len=1000, render=False, seed=1):
    env.seed(seed)
    torch.manual_seed(seed)

    def train_timestep(engine, timestep):
        state = engine.state.observation
        action, context = policy(state)
        state, reward, done, _ = env.step(action)
        policy.update(context, state, reward)
        engine.state.total_reward += reward
        if done:
            engine.terminate_epoch()
            engine.state.timestep = timestep

        return reward

    trainer = Engine(train_timestep)

    @trainer.on(Events.EPOCH_STARTED)
    def reset_environment_state(engine):
        engine.state.total_reward = 0
        engine.state.observation = env.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def update_model(engine):
        policy.end_episode()
        torch.save(policy.model, f'./saved_models/goal_model_{env.spec.id}.pkl')
        print(f'session reward: {engine.state.total_reward}')

    if render:
        @trainer.on(Events.ITERATION_COMPLETED)
        def render(_):
            env.render()
            

    pbar = ProgressBar(persist=True)
    pbar.attach(trainer)

    timesteps = list(range(episode_len))
    trainer.run(timesteps, max_epochs=n_episodes)
