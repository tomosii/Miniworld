# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 gen_dataset.py


import argparse
import gymnasium as gym
import math
import miniworld
import imageio
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing as mp

from enum import Enum
from miniworld.miniworld import MiniWorldEnv


ENV_NAME = "MiniWorld-SerialNineRooms-v0"

VIEW_MODE = "top_local"
# VIEW_MODE = "top"
# VIEW_MODE = "agent"

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

TRAIN_EPISODES = 9900
TEST_EPISODES = 100

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
        # print(image.shape)
        image = cv2.resize(
            image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
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


_WORKER_ENV = None


def _init_worker(env_name, view_mode):
    global _WORKER_ENV
    _WORKER_ENV = gym.make(env_name, render_mode="rgb_array", view=view_mode)


def _generate_and_save(args):
    split, idx, out_dir = args
    images = generate_sequence(_WORKER_ENV)
    np.savez(os.path.join(out_dir, f"{idx}.npz"), video=images)
    return split, idx


def _parse_args():
    parser = argparse.ArgumentParser(description="Generate MiniWorld dataset.")
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Number of parallel worker processes (use 1 for sequential).",
    )
    return parser.parse_args()


def main():
    args = _parse_args()
    output_root = os.path.join(os.path.dirname(__file__), "generated")
    output_dir = os.path.join(output_root, DATASET_NAME)
    os.makedirs(output_dir, exist_ok=True)

    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    # images = generate_sequence(env)
    # with imageio.get_writer(
    #     os.path.join(output_root, f"{DATASET_NAME}.gif"),
    #     mode="I",
    #     loop=0,
    #     duration=1,
    # ) as writer:
    #     for image in images:
    #         writer.append_data(image)

    # example_episodes = 6
    # example_videos = []
    # for _ in range(example_episodes):
    #     images = generate_sequence(env)
    #     example_videos.append(images)

    # min_length = min(video.shape[0] for video in example_videos)
    # plt.figure(figsize=(min_length, example_episodes + 1))
    # for i in range(example_episodes):
    #     for t in range(min_length):
    #         plt.subplot(example_episodes, min_length, i * min_length + t + 1)
    #         plt.imshow(example_videos[i][t])
    #         plt.axis("off")
    # plt.subplots_adjust(wspace=0.1, hspace=0.5)
    # plt.tight_layout()
    # plt.savefig(os.path.join(output_root, f"{DATASET_NAME}.png"))

    # return

    if args.workers == 1:
        env = gym.make(ENV_NAME, render_mode="rgb_array", view=VIEW_MODE)

        for i in range(TRAIN_EPISODES):
            print(f"\nGenerating train episode {i+1}/{TRAIN_EPISODES}")
            images = generate_sequence(env)
            np.savez(os.path.join(train_dir, f"{i}.npz"), video=images)

        print()

        for i in range(TEST_EPISODES):
            print(f"\nGenerating test episode {i+1}/{TEST_EPISODES}")
            images = generate_sequence(env)
            np.savez(os.path.join(test_dir, f"{i}.npz"), video=images)

        env.close()
    else:
        ctx = mp.get_context("spawn")
        train_jobs = [("train", i, train_dir) for i in range(TRAIN_EPISODES)]
        test_jobs = [("test", i, test_dir) for i in range(TEST_EPISODES)]
        total_jobs = len(train_jobs) + len(test_jobs)
        completed = 0

        print(
            f"Generating dataset with {args.workers} workers "
            f"({TRAIN_EPISODES} train, {TEST_EPISODES} test)"
        )

        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(ENV_NAME, VIEW_MODE),
        ) as pool:
            for split, idx in pool.imap_unordered(
                _generate_and_save, train_jobs + test_jobs, chunksize=4
            ):
                completed += 1
                if completed % 100 == 0 or completed == total_jobs:
                    print(
                        f"Completed {completed}/{total_jobs} " f"(last: {split} {idx})"
                    )

        print(f"\n✅ Created {DATASET_NAME} dataset.")


if __name__ == "__main__":
    main()
