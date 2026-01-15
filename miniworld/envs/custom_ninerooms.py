from gymnasium import spaces, utils

from miniworld.entity import Box, ImageFrame
from miniworld.miniworld import MiniWorldEnv
import math


class CustomNinerooms(MiniWorldEnv, utils.EzPickle):
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
            # wall_tex="grass",
            # no_ceiling=True,
        )
        room1 = self.add_rect_room(
            min_x=-1,
            max_x=1,
            min_z=-3,
            max_z=-1,
            floor_tex="cardboard",
            # wall_tex="cardboard",
            # no_ceiling=True,
        )
        room2 = self.add_rect_room(
            min_x=1,
            max_x=3,
            min_z=-3,
            max_z=-1,
            floor_tex="stucco",
            # wall_tex="stucco",
            # no_ceiling=True,
        )

        room3 = self.add_rect_room(
            min_x=-3,
            max_x=-1,
            min_z=-1,
            max_z=1,
            floor_tex="metal_grill",
            # wall_tex="metal_grill",
            # no_ceiling=True,
        )
        room4 = self.add_rect_room(
            min_x=-1,
            max_x=1,
            min_z=-1,
            max_z=1,
            floor_tex="water",
            # wall_tex="water",
            # no_ceiling=True,
        )
        room5 = self.add_rect_room(
            min_x=1,
            max_x=3,
            min_z=-1,
            max_z=1,
            floor_tex="wood",
            # wall_tex="wood",
            # no_ceiling=True,
        )

        room6 = self.add_rect_room(
            min_x=-3,
            max_x=-1,
            min_z=1,
            max_z=3,
            floor_tex="slime",
            # wall_tex="slime",
            # no_ceiling=True,
        )
        room7 = self.add_rect_room(
            min_x=-1,
            max_x=1,
            min_z=1,
            max_z=3,
            floor_tex="cinder_blocks",
            # wall_tex="cinder_blocks",
            # no_ceiling=True,
        )
        room8 = self.add_rect_room(
            min_x=1,
            max_x=3,
            min_z=1,
            max_z=3,
            floor_tex="lava",
            # wall_tex="lava",
            # no_ceiling=True,
        )

        self.connect_rooms(room0, room1, min_z=-2.5, max_z=-1.5)
        self.connect_rooms(room1, room2, min_z=-2.5, max_z=-1.5)

        self.connect_rooms(room3, room4, min_z=-0.5, max_z=0.5)
        self.connect_rooms(room4, room5, min_z=-0.5, max_z=0.5)

        self.connect_rooms(room6, room7, min_z=1.5, max_z=2.5)
        self.connect_rooms(room7, room8, min_z=1.5, max_z=2.5)

        self.connect_rooms(room0, room3, min_x=-2.5, max_x=-1.5)
        self.connect_rooms(room3, room6, min_x=-2.5, max_x=-1.5)

        self.connect_rooms(room1, room4, min_x=-0.5, max_x=0.5)
        self.connect_rooms(room4, room7, min_x=-0.5, max_x=0.5)

        self.connect_rooms(room2, room5, min_x=1.5, max_x=2.5)
        self.connect_rooms(room5, room8, min_x=1.5, max_x=2.5)

        self.place_agent()

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        return obs, reward, termination, truncation, info
