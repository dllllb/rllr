from gym_minigrid.envs import EmptyEnv, KeyCorridor
from gym_minigrid.minigrid import *
from gym_minigrid.register import register
from random import sample
import numpy as np


class EmptyEnv32x32(EmptyEnv):
    def __init__(self, **kwargs):
        super().__init__(size=32, **kwargs)


class EmptyEnvRandom(EmptyEnv):
    def __init__(self,
                 size=8,
                 agent_start_pos=None,
                 agent_start_dir=0):

        super(EmptyEnvRandom, self).__init__(size, agent_start_pos, agent_start_dir)

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place a goal square in the bottom-right corner
        self.place_obj(Goal())

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"


class KeyCorridorNoSameColors(KeyCorridor):
    def __init__(self, num_rows=3, obj_type="ball", room_size=6, seed=None):
        self.door_object_ind = OBJECT_TO_IDX['door']
        self.empty_object_ind = OBJECT_TO_IDX['empty']
        super().__init__(num_rows, obj_type, room_size, seed)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)
        grid = self.unwrapped.grid.encode()
        obj_x, obj_y = self.obj.cur_pos
        yy, xx = np.where(grid[:, :, 0] == self.door_object_ind)
        colors = grid[yy, xx, 1]
        states = grid[yy, xx, 2]

        closed_door_ind, = np.where(states == STATE_TO_IDX['locked'])
        closed_door_color = colors[closed_door_ind]
        new_xx, new_yy = [int(xx[closed_door_ind])], [int(yy[closed_door_ind])]
        new_colors, new_states = [int(closed_door_color)], [int(states[closed_door_ind])]

        other_doors_ind, = np.where(states != STATE_TO_IDX['locked'])
        other_doors_colors = [idx for idx in COLOR_TO_IDX.values() if idx not in closed_door_color]
        for door_ind in np.random.permutation(other_doors_ind)[:len(other_doors_colors)]:
            new_xx.append(xx[door_ind])
            new_yy.append(yy[door_ind])
            new_colors.append(other_doors_colors.pop(-1))
            new_states.append(states[door_ind])

        new_xx = np.array(new_xx)
        new_yy = np.array(new_yy)

        grid[yy, xx, 0] = self.empty_object_ind
        grid[yy, xx, 1:] = 0

        grid[new_yy, new_xx, 0] = self.door_object_ind
        grid[new_yy, new_xx, 1] = new_colors
        grid[new_yy, new_xx, 2] = new_states

        self.unwrapped.grid, _ = self.unwrapped.grid.decode(grid)
        self.obj = self.unwrapped.grid.get(obj_x, obj_y)


class HardKeyCorridor(KeyCorridorNoSameColors):
    def __init__(self, num_rows=3, obj_type="ball", room_size=6, seed=None):
        super().__init__(num_rows, obj_type, room_size, seed)

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.balls = [self.obj]
        balls_colors = [self.obj.color]
        available_balls_colors = set(COLOR_TO_IDX)
        available_balls_colors.remove(balls_colors[0])

        self.boxes = list()
        boxes_colors = list()
        available_boxes_colors = set(COLOR_TO_IDX)

        # add 1 box
        while True:
            try:
                i = np.random.choice([0, 2])
                j = np.random.randint(0, self.num_rows)
                c = sample(available_boxes_colors, 1)[0]
                new_box, _ = self.add_object(i, j, kind='box', color=c)
                available_boxes_colors.remove(c)
                self.boxes.append(new_box)
            except:
                pass
            else:
                break

        # add 2 more balls
        for _ in range(2):
            i = np.random.choice([0, 2])
            j = np.random.randint(0, self.num_rows)
            c = sample(available_balls_colors, 1)[0]
            try:
                new_ball, _ = self.add_object(i, j, kind='ball', color=c)
                available_balls_colors.remove(c)
                self.balls.append(new_ball)
            except:
                pass

        # add 2 more boxes
        for _ in range(2):
            i = np.random.choice([0, 2])
            j = np.random.randint(0, self.num_rows)
            c = sample(available_boxes_colors, 1)[0]
            try:
                new_box, _ = self.add_object(i, j, kind='box', color=c)
                available_boxes_colors.remove(c)
                self.boxes.append(new_box)
            except:
                pass

        self.grab_box = False
        self.grab_ball = False

    def step(self, action):
        obs, reward, done, info = super().step(action)
        if done and reward > 0:
            done = False
            reward = 0

        if action == self.actions.pickup:
            if not self.grab_box:
                if self.carrying and self.carrying.type == self.boxes[0].type:
                    self.grab_box = True
            else:
                if self.carrying and self.carrying.type == self.balls[0].type:
                    reward = self._reward()
                    done = True
                    self.grab_ball = True

        return obs, reward, done, info


class KeyCorridorNoSameColorsS5R3(KeyCorridorNoSameColors):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=3,
            seed=seed
        )


class KeyCorridorNoSameColorsS4R3(KeyCorridorNoSameColors):
    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed
        )


class KeyCorridorNoSameColorsS3R3(KeyCorridorNoSameColors):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=3,
            seed=seed
        )


class HardKeyCorridorS5R3(HardKeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=5,
            num_rows=3,
            seed=seed
        )


class HardKeyCorridorS4R3(HardKeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed
        )


class HardKeyCorridorS3R3(HardKeyCorridor):
    def __init__(self, seed=None):
        super().__init__(
            room_size=3,
            num_rows=3,
            seed=seed
        )


register(id='MiniGrid-Empty-32x32-v0',
         entry_point='rllr.env.custom_envs:EmptyEnv32x32')

register(id='MiniGrid-Empty-RandomGoal-8x8-v0',
         entry_point='rllr.env.custom_envs:EmptyEnvRandom')

register(id='MiniGrid-KeyCorridorNoSameColorsS6R3-v0',
         entry_point='rllr.env.custom_envs:KeyCorridorNoSameColors')

register(id='MiniGrid-KeyCorridorNoSameColorsS5R3-v0',
         entry_point='rllr.env.custom_envs:KeyCorridorNoSameColorsS5R3')

register(id='MiniGrid-KeyCorridorNoSameColorsS4R3-v0',
         entry_point='rllr.env.custom_envs:KeyCorridorNoSameColorsS4R3')

register(id='MiniGrid-KeyCorridorNoSameColorsS3R3-v0',
         entry_point='rllr.env.custom_envs:KeyCorridorNoSameColorsS3R3')

register(id='MiniGrid-HardKeyCorridorS6R3-v0',
         entry_point='rllr.env.custom_envs:HardKeyCorridor')

register(id='MiniGrid-HardKeyCorridorS5R3-v0',
         entry_point='rllr.env.custom_envs:HardKeyCorridorS5R3')

register(id='MiniGrid-HardKeyCorridorS4R3-v0',
         entry_point='rllr.env.custom_envs:HardKeyCorridorS4R3')

register(id='MiniGrid-HardKeyCorridorS3R3-v0',
         entry_point='rllr.env.custom_envs:HardKeyCorridorS3R3')
