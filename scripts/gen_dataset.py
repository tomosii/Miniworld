import gymnasium as gym
import math
import miniworld
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt

from enum import Enum
from miniworld.miniworld import MiniWorldEnv


ENV_NAME = "MiniWorld-SerialNineRooms-v0"

VIEW_MODE = "top_local"
# VIEW_MODE = "top"
# VIEW_MODE = "agent"

TRAIN_EPISODES = 10
TEST_EPISODES = 10

DATASET_NAME = "serial_nine_rooms"


class Phase(Enum):
    A_VERTICAL = 0
    A_TURN = 1

    B_HORIZONTAL = 2
    B_TURN = 3

    C_VERTICAL = 4
    C_TURN = 5

    D_HORIZONTAL = 6
    D_TURN = 7

    E_VERTICAL = 8


# turn_left = 0
# turn_right = 1
# move_forward = 2
# move_back = 3


def match_position(value1, value2):
    return abs(value1 - value2) < 0.1


def match_direction(value1, value2):
    # Allow 10 degree of error
    return abs(value1 - value2) < 10 * (math.pi / 180)


def generate_sequence(env):
    env.reset()

    room_size = env.unwrapped.ROOM_SIZE
    env_edge = env.unwrapped.ENV_EDGE

    step = 0
    phase = Phase.A_VERTICAL
    action = MiniWorldEnv.Actions.move_forward

    images = []

    while True:
        if phase == Phase.A_VERTICAL:
            action = MiniWorldEnv.Actions.move_forward
        elif phase == Phase.A_TURN:
            action = MiniWorldEnv.Actions.turn_left
        elif phase == Phase.B_HORIZONTAL:
            action = MiniWorldEnv.Actions.move_forward
        elif phase == Phase.B_TURN:
            action = MiniWorldEnv.Actions.turn_left
        elif phase == Phase.C_VERTICAL:
            action = MiniWorldEnv.Actions.move_forward
        elif phase == Phase.C_TURN:
            action = MiniWorldEnv.Actions.turn_right
        elif phase == Phase.D_HORIZONTAL:
            action = MiniWorldEnv.Actions.move_forward
        elif phase == Phase.D_TURN:
            action = MiniWorldEnv.Actions.turn_right
        elif phase == Phase.E_VERTICAL:
            action = MiniWorldEnv.Actions.move_forward

        env.step(action)
        image = env.render()
        images.append(image)

        pos = env.unwrapped.agent.pos
        dir = env.unwrapped.agent.dir

        if phase == Phase.A_VERTICAL and match_position(
            pos[2], env_edge - (room_size / 2)
        ):
            phase = Phase.A_TURN

        if phase == Phase.A_TURN and match_direction(dir, 0):
            phase = Phase.B_HORIZONTAL

        if phase == Phase.B_HORIZONTAL and match_position(pos[0], 0):
            phase = Phase.B_TURN

        if phase == Phase.B_TURN and match_direction(dir, math.pi / 2):
            phase = Phase.C_VERTICAL

        if phase == Phase.C_VERTICAL and match_position(
            pos[2], -env_edge + (room_size / 2)
        ):
            phase = Phase.C_TURN

        if phase == Phase.C_TURN and match_direction(dir, 0):
            phase = Phase.D_HORIZONTAL

        if phase == Phase.D_HORIZONTAL and match_position(
            pos[0], env_edge - (room_size / 2)
        ):
            phase = Phase.D_TURN

        if phase == Phase.D_TURN and match_direction(dir, -math.pi / 2):
            phase = Phase.E_VERTICAL

        if phase == Phase.E_VERTICAL and match_position(
            pos[2], env_edge - (room_size / 2)
        ):
            # Episode complete
            break

        step += 1

    print(f"Episode ended after {step} steps")

    return np.array(images, dtype=np.uint8)


def main():
    output_root = os.path.join(os.path.dirname(__file__), "generated")
    output_dir = os.path.join(output_root, DATASET_NAME)
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    env = gym.make(ENV_NAME, render_mode="rgb_array", view=VIEW_MODE)

    images = generate_sequence(env)
    with imageio.get_writer(
        os.path.join(output_root, f"{DATASET_NAME}.gif"),
        mode="I",
        loop=0,
        duration=1,
    ) as writer:
        for image in images:
            writer.append_data(image)

    example_episodes = 6
    example_videos = []
    for _ in range(example_episodes):
        images = generate_sequence(env)
        example_videos.append(images)

    min_length = min(video.shape[0] for video in example_videos)
    plt.figure(figsize=(min_length, example_episodes + 1))
    for i in range(example_episodes):
        for t in range(min_length):
            plt.subplot(example_episodes, min_length, i * min_length + t + 1)
            plt.imshow(example_videos[i][t])
            plt.axis("off")
    plt.subplots_adjust(wspace=0.1, hspace=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_root, f"{DATASET_NAME}.png"))

    for i in range(TRAIN_EPISODES):
        print(f"\nGenerating train episode {i+1}/{TRAIN_EPISODES}")
        images = generate_sequence(env)
        np.savez(os.path.join(train_dir, f"{i}.npz"), video=images)

    print()

    for i in range(TEST_EPISODES):
        print(f"\nGenerating test episode {i+1}/{TEST_EPISODES}")
        images = generate_sequence(env)
        np.savez(os.path.join(test_dir, f"{i}.npz"), video=images)

    print(f"\n✅ Created {DATASET_NAME} dataset.")

    env.close()


if __name__ == "__main__":
    main()
