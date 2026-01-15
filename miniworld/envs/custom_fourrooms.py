from gymnasium import spaces, utils

from miniworld.entity import Box, ImageFrame
from miniworld.miniworld import MiniWorldEnv
import math


class CustomFourRooms(MiniWorldEnv, utils.EzPickle):
    """
    ## Description

    Classic four rooms environment. The goal is to reach the red box to get a
    reward in as few steps as possible.

    ## Action Space

    | Num | Action                      |
    |-----|-----------------------------|
    | 0   | turn left                   |
    | 1   | turn right                  |
    | 2   | move forward                |

    ## Observation Space

    The observation space is an `ndarray` with shape `(obs_height, obs_width, 3)`
    representing an RGB image of what the agents see.

    ## Rewards

    +(1 - 0.2 * (step_count / max_episode_steps)) when red box reached and zero otherwise.

    ## Arguments

    ```python
    env = gymnasium.make("MiniWorld-FourRooms-v0")
    ```

    """

    def __init__(self, **kwargs):
        MiniWorldEnv.__init__(self, max_episode_steps=250, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):
        room0 = self.add_rect_room(
            min_x=-3,
            max_x=-1,
            min_z=-3,
            max_z=-1,
            floor_tex="grass",
        )

        # Top-left room
        room0 = self.add_rect_room(
            min_x=-2,
            max_x=0,
            min_z=0,
            max_z=2,
            wall_tex="brick_wall",
            floor_tex="grass",
            # no_ceiling=True,
        )
        # Top-right room
        room1 = self.add_rect_room(
            min_x=0,
            max_x=2,
            min_z=0,
            max_z=2,
            # wall_tex="brick_wall",
            floor_tex="wood",
            # no_ceiling=True,
        )
        # Bottom-right room
        room2 = self.add_rect_room(
            min_x=0,
            max_x=2,
            min_z=-2,
            max_z=0,
            # wall_tex="brick_wall",
            floor_tex="water",
            # no_ceiling=True,
        )
        # Bottom-left room
        room3 = self.add_rect_room(
            min_x=-2,
            max_x=0,
            min_z=-2,
            max_z=0,
            # wall_tex="brick_wall",
            floor_tex="lava",
            # no_ceiling=True,
        )

        # Add openings to connect the rooms together
        self.connect_rooms(room0, room1, min_z=0.5, max_z=1.5)
        self.connect_rooms(room1, room2, min_x=0.5, max_x=1.5)
        self.connect_rooms(room2, room3, min_z=-1.5, max_z=-0.5)
        self.connect_rooms(room3, room0, min_x=-1.5, max_x=-0.5)
        # self.connect_rooms(room2, room3, min_z=0.5, max_z=1.5)
        # self.connect_rooms(room3, room0, min_x=0.5, max_x=1.5)
        # self.connect_rooms(room1, room2, min_x=3, max_x=5, max_y=2.2)
        # self.connect_rooms(room2, room3, min_z=-5, max_z=-3, max_y=2.2)
        # self.connect_rooms(room3, room0, min_x=-5, max_x=-3, max_y=2.2)

        # self.box = self.place_entity(Box(color="red"))

        # Door visual on room0 east wall, facing into room0
        # door = ImageFrame(
        #     pos=[-0.01, 1.0, 0.5],
        #     dir=math.pi,
        #     tex_name="door_steel_brown",
        #     width=1.0,
        # )
        # self.place_entity(door, pos=door.pos, dir=door.dir)

        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # if self.near(self.box):
        #     reward += self._reward()
        #     termination = True

        return obs, reward, termination, truncation, info
