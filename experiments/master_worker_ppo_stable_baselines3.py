import train_worker
from master_worker_dqn_stable_baselines3 import GoalStateExtenderWrapper
from master_worker_dqn_stable_baselines3 import ExtendedStateFeatureExtractor, StateExtenderWrapper

from rllr.env.gym_minigrid_navigation import environments as minigrid_envs

from stable_baselines3 import PPO


def train_1st_stage():

    worker_env_config = {
        "env_type": "gym_minigrid",
        "env_task": "MiniGrid-Empty",
        "grid_size": 8,
        "action_size": 3,
        "rgb_image": False,
        "goal_achieving_criterion": "position",
        "goal_type": "random",
        "video_path": "artifacts/video/"
    }

    worker_env = train_worker.gen_navigation_env(worker_env_config)
    worker_env = GoalStateExtenderWrapper(worker_env)

    policy_kwargs = dict(
        features_extractor_class=ExtendedStateFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=64, hidden=128),
        net_arch=[128, 128]
    )

    agent_kwargs = {
        "learning_rate": 1e-3,
        "batch_size": 128,
        "gamma": 0.9,
        "seed": 0,
        "policy_kwargs": policy_kwargs,
        "verbose": 1
    }

    agent = PPO("MlpPolicy", worker_env, **agent_kwargs)
    agent.learn(100000)

    return agent


def train_2nd_stage(agent):
    master_env_config = {
        "env_type": "gym_minigrid",
        "env_task": "MiniGrid-Empty",
        "grid_size": 8,
        "action_size": 3,
        "rgb_image": False,
        "video_path": "artifacts/video/"
    }

    master_env = minigrid_envs.gen_wrapped_env(master_env_config)
    master_env = StateExtenderWrapper(master_env)
    agent.set_env(master_env)

    for param in agent.policy.mlp_extractor.parameters():
        param.requires_gradient = False

    for param in agent.policy.features_extractor.state_encoder.parameters():
        param.requires_gradient = False

    for param in agent.policy.action_net.parameters():
        param.requires_gradient = False

    for param in agent.policy.value_net.parameters():
        param.requires_gradient = False

    agent.learn(100000, log_interval=10)


if __name__ == '__main__':
    agent = train_1st_stage()
    train_2nd_stage(agent)
