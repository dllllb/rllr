from gym_minigrid.minigrid import *
from gym_minigrid.register import register
import random

class MyEmptyEnv(MiniGridEnv):
    """
    Empty grid environment, no obstacles, sparse reward
    """

    def __init__(
        self,
        size=8,
        agent_start_pos=(1,1),
        agent_start_dir=0,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        super().__init__(
            grid_size=size,
            max_steps=10*size*size,
            # Set this to True for maximum speed
            see_through_walls=True,
            agent_view_size=size
        )

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        #self.put_obj(Goal(), width - 2, height - 2)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

class MyEmptyRandomPosEnv(MyEmptyEnv):


    def __init__(self,
                 size=20,
                 agent_start_dir=0):
        agent_start_pos = None#(random.randint(1, size-1), random.randint(1, size-1))
        super().__init__(size, agent_start_pos, agent_start_dir)



register(
    id='MiniGrid-MyEmpty-8x8-v0',
    entry_point='custom_envs:MyEmptyEnv'
)

register(
    id='MiniGrid-MyEmptyRandomPos-8x8-v0',
    entry_point='custom_envs:MyEmptyRandomPosEnv'
)