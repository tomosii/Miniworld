from gymnasium import spaces, utils

from miniworld.entity import Box, ImageFrame
from miniworld.miniworld import MiniWorldEnv
import math


class SerialNineRooms(MiniWorldEnv, utils.EzPickle):
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

    ROOM_SIZE = 2
    HALLWAY_LENGTH = 2
    HALLWAY_WIDTH = 1
    HALLWAY_MARGIN = (ROOM_SIZE - HALLWAY_WIDTH) / 2

    ENV_SIZE = ROOM_SIZE * 3 + HALLWAY_LENGTH * 2
    ENV_EDGE = ENV_SIZE / 2

    TEXTURES = [
        "grass",
        "cardboard",
        "stucco",
        "metal_grill",
        "water",
        "wood",
        "slime",
        "rock",
        "lg_style_05_wall_yellow_d_result",
        # "lava",
    ]

    def __init__(self, **kwargs):
        MiniWorldEnv.__init__(self, max_episode_steps=250, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _gen_world(self):

        rooms = []

        # Add rooms
        for row in range(3):
            for col in range(3):
                room = self.add_rect_room(
                    min_x=-self.ENV_EDGE + col * (self.ROOM_SIZE + self.HALLWAY_LENGTH),
                    max_x=-self.ENV_EDGE
                    + col * (self.ROOM_SIZE + self.HALLWAY_LENGTH)
                    + self.ROOM_SIZE,
                    min_z=-self.ENV_EDGE + row * (self.ROOM_SIZE + self.HALLWAY_LENGTH),
                    max_z=-self.ENV_EDGE
                    + row * (self.ROOM_SIZE + self.HALLWAY_LENGTH)
                    + self.ROOM_SIZE,
                    floor_tex=self.TEXTURES[row * 3 + col],
                )
                rooms.append(room)
                # print(
                #     f"min_x: {room.min_x}, max_x: {room.max_x}, min_z: {room.min_z}, max_z: {room.max_z}"
                # )

        for i, room in enumerate(rooms):
            print(
                f"Room {i}: min_x: {room.min_x}, max_x: {room.max_x}, min_z: {room.min_z}, max_z: {room.max_z}"
            )

        # Add hallways between rooms (only vertical)
        for col in range(3):
            for row in range(2):
                hallway = self.add_rect_room(
                    min_x=-self.ENV_EDGE
                    + self.HALLWAY_MARGIN
                    + col * (self.ROOM_SIZE + self.HALLWAY_LENGTH),
                    max_x=-self.ENV_EDGE
                    + self.HALLWAY_MARGIN
                    + col * (self.ROOM_SIZE + self.HALLWAY_LENGTH)
                    + self.HALLWAY_WIDTH,
                    min_z=-self.ENV_EDGE
                    + self.ROOM_SIZE
                    + row * (self.ROOM_SIZE + self.HALLWAY_LENGTH),
                    max_z=-self.ENV_EDGE
                    + self.ROOM_SIZE
                    + row * (self.ROOM_SIZE + self.HALLWAY_LENGTH)
                    + self.HALLWAY_LENGTH,
                    floor_tex="cinder_blocks",
                )

                # connect hallway to rooms
                room_before = rooms[row * 3 + col]
                room_after = rooms[(row + 1) * 3 + col]
                self.connect_rooms(
                    hallway,
                    room_before,
                    min_x=-self.ENV_EDGE
                    + self.HALLWAY_MARGIN
                    + col * (self.ROOM_SIZE + self.HALLWAY_LENGTH),
                    max_x=-self.ENV_EDGE
                    + self.HALLWAY_MARGIN
                    + col * (self.ROOM_SIZE + self.HALLWAY_LENGTH)
                    + self.HALLWAY_WIDTH,
                )
                self.connect_rooms(
                    hallway,
                    room_after,
                    min_x=-self.ENV_EDGE
                    + self.HALLWAY_MARGIN
                    + col * (self.ROOM_SIZE + self.HALLWAY_LENGTH),
                    max_x=-self.ENV_EDGE
                    + self.HALLWAY_MARGIN
                    + col * (self.ROOM_SIZE + self.HALLWAY_LENGTH)
                    + self.HALLWAY_WIDTH,
                )

        # Add two horizontal hallways
        hallway_horizontal_bottom_right = self.add_rect_room(
            min_x=-self.ENV_EDGE + self.ROOM_SIZE,
            max_x=-self.ENV_EDGE + self.ROOM_SIZE + self.HALLWAY_LENGTH,
            min_z=self.ENV_EDGE - self.ROOM_SIZE + self.HALLWAY_MARGIN,
            max_z=self.ENV_EDGE - self.HALLWAY_MARGIN,
            floor_tex="cinder_blocks",
        )
        self.connect_rooms(
            hallway_horizontal_bottom_right,
            rooms[6],
            min_z=self.ENV_EDGE - self.ROOM_SIZE + self.HALLWAY_MARGIN,
            max_z=self.ENV_EDGE - self.HALLWAY_MARGIN,
        )
        self.connect_rooms(
            hallway_horizontal_bottom_right,
            rooms[7],
            min_z=self.ENV_EDGE - self.ROOM_SIZE + self.HALLWAY_MARGIN,
            max_z=self.ENV_EDGE - self.HALLWAY_MARGIN,
        )

        hallway_horizontal_top_left = self.add_rect_room(
            min_x=self.ENV_EDGE - self.ROOM_SIZE - self.HALLWAY_LENGTH,
            max_x=self.ENV_EDGE - self.ROOM_SIZE,
            min_z=-self.ENV_EDGE + self.HALLWAY_MARGIN,
            max_z=-self.ENV_EDGE + self.HALLWAY_MARGIN + self.HALLWAY_WIDTH,
            floor_tex="cinder_blocks",
        )
        self.connect_rooms(
            hallway_horizontal_top_left,
            rooms[1],
            min_z=-self.ENV_EDGE + self.HALLWAY_MARGIN,
            max_z=-self.ENV_EDGE + self.HALLWAY_MARGIN + self.HALLWAY_WIDTH,
        )
        self.connect_rooms(
            hallway_horizontal_top_left,
            rooms[2],
            min_z=-self.ENV_EDGE + self.HALLWAY_MARGIN,
            max_z=-self.ENV_EDGE + self.HALLWAY_MARGIN + self.HALLWAY_WIDTH,
        )

        self.place_agent(
            # room=room0,
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        return obs, reward, termination, truncation, info
