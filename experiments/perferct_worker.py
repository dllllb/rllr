import logging
from rllr.models.encoders import GoalStateEncoder
from torch import nn
import torch
import numpy as np
from pathlib import Path
from rllr.utils.logger import init_logger
from rllr.utils import convert_to_torch

from experiments import train_worker, train_master
from rllr.utils import get_conf, switch_reproducibility_on

logger = logging.getLogger(__name__)


class StatePosEncoder(nn.Module):
    """
    code minigrid raw states as agent position and direction via one hot encoder
    """
    def __init__(self, grid_size, is_goal):
        super().__init__()
        self.output_size = (grid_size - 2) ** 2 + 4
        self.grid_size = grid_size
        self.is_goal = is_goal

    def forward(self, state):
        x = state[0, 1:-1, 1:-1].numpy()
        agent_pos = np.where(x == 10)[:-1]
        direction = x[agent_pos]
        if not self.is_goal:
            return torch.tensor([[(agent_pos[0][0]/(self.grid_size-2)-0.5)*2,
                                  (agent_pos[1][0]/(self.grid_size-2)-0.5)*2,
                                  direction[0][-1]]])
        else:
            return torch.tensor([[(agent_pos[0][0] / (self.grid_size - 2)-0.5)*2,
                                  (agent_pos[1][0] / (self.grid_size - 2)-0.5)*2]])


class PerfectWorker():

    def __init__(self, state_encoder):
        self.state_encoder = state_encoder
        self.device = "cpu"

    def update(self, state, action, reward, next_state, done):
        return

    def reset_episode(self):
        return

    def act(self, state):
        """
        Directions:
        0 >
        1 V
        2 <
        3 ^
        Actions:
        0 - counter clockwise
        1 - clockwise
        2 - straight
        """
        state = convert_to_torch([state], self.device)
        pos_and_dir = self.state_encoder.forward(state)
        agent_pos_x = pos_and_dir[0][0]
        agent_pos_y = pos_and_dir[0][1]
        agent_dir = pos_and_dir[0][2]
        goal_pos_x = pos_and_dir[0][3]
        goal_pos_y = pos_and_dir[0][4]
        action = None
        if agent_pos_x < goal_pos_x: # Agent left from goal
            if agent_dir == 0: # Go to the right towards goal
                action = 2
            elif agent_dir == 1:
                if agent_pos_y < goal_pos_y: # Agent up from goal
                    action = 2
                else:
                    action = 0
            elif agent_dir == 2:
                if agent_pos_y < goal_pos_y:
                    action = 0
                else:
                    action = 1
            elif agent_dir == 3:
                if agent_pos_y < goal_pos_y:
                    action = 1
                else:
                    action = 2
        else: # Agent right from goal
            if agent_dir == 0:  # Go to the right towards goal
                if agent_pos_y < goal_pos_y:
                    action = 1
                else:
                    action = 0
            elif agent_dir == 1:
                if agent_pos_y < goal_pos_y:
                    action = 2
                else:
                    action = 1
            elif agent_dir == 2:
                action = 2
            elif agent_dir == 3:
                if agent_pos_y < goal_pos_y:
                    action = 0
                else:
                    action = 2
        return action


def main():
    init_logger("experiments.train_worker")
    conf = get_conf([f"-c{Path(__file__).parent.absolute()}/conf/perfect_worker.hocon"])
    switch_reproducibility_on(conf['seed'])
    env = train_worker.gen_navigation_env(conf["env"])

    state_encoder = StatePosEncoder(grid_size=conf["env"]["grid_size"], is_goal=False)
    goal_state_encoder = StatePosEncoder(grid_size=conf["env"]["grid_size"], is_goal=True)
    encoder = GoalStateEncoder(state_encoder=state_encoder, goal_state_encoder=goal_state_encoder)
    perfect_worker = PerfectWorker(encoder)
    scores, steps = train_worker.train_worker(env, perfect_worker, n_episodes=100, verbose=10)

    # Check worker
    print(f"Maximum steps: {max(steps)}, , average_steps: {sum(steps)/len(steps)}, minimum score: {min(scores): .2f}")

    # Train master agent
    init_logger("experiments.train_master")
    del conf['env']['goal_achieving_criterion']
    del conf['env']['goal_type']
    env = train_worker.gen_env(conf['env'])

    master = train_master.get_master_agent(conf["master"]["emb_size"], conf)
    train_master.train_master(env, perfect_worker, master,
                              n_episodes=conf['training']['n_episodes'],
                              verbose=conf['training']['verbose'], worker_steps=1)


if __name__ == '__main__':
    main()

