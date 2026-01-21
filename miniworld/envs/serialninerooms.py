from gymnasium import spaces, utils

from miniworld.entity import Box, ImageFrame
from miniworld.miniworld import MiniWorldEnv
import math
import random


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

    ROOM_SIZE = 2.4
    HALLWAY_LENGTH = ROOM_SIZE * 1.5
    HALLWAY_WIDTH = ROOM_SIZE * 0.5
    HALLWAY_MARGIN = (ROOM_SIZE - HALLWAY_WIDTH) / 2

    ENV_SIZE = ROOM_SIZE * 3 + HALLWAY_LENGTH * 2
    ENV_EDGE = ENV_SIZE / 2

    TEXTURES = [
        "grass_1",
        "water_1",
        "wood_1",
        "slime_1",
        "lava_1",
        "floor_tiles_bw_1",
        "lg_style_01_4tile_d_result",
        "lg_style_01_wall_blue_1",
        "lg_style_02_wall_dblue_d_result",
        "lg_style_02_wall_purple_d_result",
        "lg_style_03_wall_light_m_result",
        "lg_style_03_wall_orange_1",
        "lg_style_03_wall_purple_d_result",
        "lg_style_04_wall_cerise_d_result",
        "lg_style_04_wall_purple_d_result",
        "lg_style_05_floor_blue_bright_d_result",
        "lg_style_05_wall_yellow_bright_d_result",
        "lg_style_04_floor_cyan_d_result",
        "lg_style_01_wall_red_1",
        "lg_style_02_floor_green_d_result",
    ]

    TEXTURE_MAPPING = [
        12,
        9,
        16,
        0,
        18,
        6,
        10,
        19,
        15,
        7,
        1,
        17,
        3,
        11,
        4,
        14,
        2,
        5,
        13,
        8,
    ]

    TEXTURE_DEPENDENT_LENGTH = 3

    def __init__(self, **kwargs):
        MiniWorldEnv.__init__(self, max_episode_steps=1000, **kwargs)
        utils.EzPickle.__init__(self, **kwargs)

        # Allow only the movement actions
        self.action_space = spaces.Discrete(self.actions.move_forward + 1)

    def _get_mapped_texture(self, texture_name):
        current_index = self.TEXTURES.index(texture_name)
        return self.TEXTURES[self.TEXTURE_MAPPING[current_index]]

    def _get_shuffled_textures(self):
        texture_list = random.sample(self.TEXTURES, self.TEXTURE_DEPENDENT_LENGTH)
        # print(f"Initial texture list: {texture_list}")
        for _ in range(9):
            texture = self._get_mapped_texture(
                texture_list[-self.TEXTURE_DEPENDENT_LENGTH]
            )
            texture_list.append(texture)
        return texture_list[self.TEXTURE_DEPENDENT_LENGTH :]

    def _gen_world(self):
        # Create rooms
        rooms = []
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
                    # floor_tex=self.TEXTURES[i],
                    # floor_tex=current_texture,
                )
                rooms.append(room)
                # print(
                #     f"min_x: {room.min_x}, max_x: {room.max_x}, min_z: {room.min_z}, max_z: {room.max_z}"
                # )

        # for i, room in enumerate(rooms):
        #     # room.floor_tex_name = shuffled_textures[i]
        #     print(
        #         f"Room {i}: min_x: {room.min_x}, max_x: {room.max_x}, min_z: {room.min_z}, max_z: {room.max_z}, floor_tex: {room.floor_tex_name}"
        #     )

        # Assign textures to rooms
        shuffled_textures = self._get_shuffled_textures()
        # print(f"Shuffled textures: {shuffled_textures}")
        rooms[0].floor_tex_name = shuffled_textures[0]
        rooms[1].floor_tex_name = shuffled_textures[5]
        rooms[2].floor_tex_name = shuffled_textures[6]
        rooms[3].floor_tex_name = shuffled_textures[1]
        rooms[4].floor_tex_name = shuffled_textures[4]
        rooms[5].floor_tex_name = shuffled_textures[7]
        rooms[6].floor_tex_name = shuffled_textures[2]
        rooms[7].floor_tex_name = shuffled_textures[3]
        rooms[8].floor_tex_name = shuffled_textures[8]

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
            # room=rooms[0],
            pos=[
                -self.ENV_EDGE + self.ROOM_SIZE / 2,
                0,
                -self.ENV_EDGE + self.ROOM_SIZE / 2,
            ],
            dir=-math.pi / 2,
        )

    def step(self, action):
        obs, reward, termination, truncation, info = super().step(action)

        # print(f"Agent position: {self.agent.pos}, direction: {self.agent.dir}")

        return obs, reward, termination, truncation, info
