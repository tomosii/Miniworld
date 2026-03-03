# xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" python3 gen_dataset.py


import argparse
import imageio
import gymnasium as gym
import miniworld
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing as mp

from miniworld.miniworld import MiniWorldEnv


ENV_NAME = "MiniWorld-MeshNineRooms-v0"

VIEW_MODE = "top_local"
# VIEW_MODE = "top"
# VIEW_MODE = "agent"

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64

TRAIN_EPISODES = 9900
TEST_EPISODES = 100

MAX_EPISODE_STEPS = 200

DATASET_NAME = "mesh_nine_rooms"


# turn_left = 0
# turn_right = 1
# move_forward = 2
# move_back = 3


def _normalize_angle(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _is_angle_close(current, target, tolerance_rad=0.12):
    return abs(_normalize_angle(target - current)) < tolerance_rad


def _is_pos_close(pos, target_x, target_z, tolerance=0.1):
    return abs(pos[0] - target_x) < tolerance and abs(pos[2] - target_z) < tolerance


def _room_centers(env):
    room_size = env.unwrapped.ROOM_SIZE
    env_edge = env.unwrapped.ENV_EDGE
    hallway_length = env.unwrapped.HALLWAY_LENGTH
    pitch = room_size + hallway_length

    centers = {}
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            centers[idx] = (
                -env_edge + (room_size / 2) + col * pitch,
                -env_edge + (room_size / 2) + row * pitch,
            )
    return centers


def _room_adjacency():
    adjacency = {idx: [] for idx in range(9)}
    for row in range(3):
        for col in range(3):
            idx = row * 3 + col
            if row > 0:
                adjacency[idx].append((row - 1) * 3 + col)
            if row < 2:
                adjacency[idx].append((row + 1) * 3 + col)
            if col > 0:
                adjacency[idx].append(row * 3 + (col - 1))
            if col < 2:
                adjacency[idx].append(row * 3 + (col + 1))
    return adjacency


def _desired_direction(src_center, dst_center):
    src_x, src_z = src_center
    dst_x, dst_z = dst_center
    dx = dst_x - src_x
    dz = dst_z - src_z

    if abs(dx) > abs(dz):
        return 0.0 if dx > 0 else np.pi
    return -np.pi / 2 if dz > 0 else np.pi / 2


def _episode_seed(base_seed, episode_idx):
    if base_seed is None:
        return int.from_bytes(os.urandom(8), "little")
    return int(base_seed) + int(episode_idx)


def generate_sequence(env, seed=None):
    rng = np.random.default_rng(seed)
    env.reset(seed=seed)

    room_centers = _room_centers(env)
    adjacency = _room_adjacency()

    images = []
    current_room = int(rng.integers(0, 9))
    start_x, start_z = room_centers[current_room]
    # Start each episode at the center of a random room.
    env.unwrapped.agent.pos = np.array([start_x, env.unwrapped.agent.pos[1], start_z])
    # Choose first target and orient the agent so the first action can be forward.
    target_room = int(rng.choice(adjacency[current_room]))
    desired_dir = _desired_direction(
        room_centers[current_room], room_centers[target_room]
    )
    env.unwrapped.agent.dir = desired_dir
    mode = "move_forward"

    for step in range(MAX_EPISODE_STEPS):
        pos = env.unwrapped.agent.pos
        direction = env.unwrapped.agent.dir

        if mode == "choose_target":
            neighbors = adjacency[current_room]
            target_room = int(rng.choice(neighbors))
            desired_dir = _desired_direction(
                room_centers[current_room], room_centers[target_room]
            )
            mode = "turn"

        if mode == "turn":
            if _is_angle_close(direction, desired_dir):
                action = MiniWorldEnv.Actions.move_forward
                mode = "move_forward"
            else:
                delta = _normalize_angle(desired_dir - direction)
                action = (
                    MiniWorldEnv.Actions.turn_left
                    if delta > 0
                    else MiniWorldEnv.Actions.turn_right
                )
        else:
            action = MiniWorldEnv.Actions.move_forward

        env.step(action)
        image = env.render()
        image = cv2.resize(
            image, (IMAGE_WIDTH, IMAGE_HEIGHT), interpolation=cv2.INTER_LINEAR
        )
        images.append(image)

        if mode == "move_forward":
            pos = env.unwrapped.agent.pos
            target_x, target_z = room_centers[target_room]
            if _is_pos_close(pos, target_x, target_z):
                current_room = target_room
                mode = "choose_target"

    print(f"Episode ended after {MAX_EPISODE_STEPS} steps")

    return np.array(images, dtype=np.uint8)


_WORKER_ENV = None


def _init_worker(env_name, view_mode):
    global _WORKER_ENV
    _WORKER_ENV = gym.make(env_name, render_mode="rgb_array", view=view_mode)


def _generate_and_save(args):
    split, idx, out_dir, base_seed = args
    seed = _episode_seed(base_seed, idx)
    images = generate_sequence(_WORKER_ENV, seed=seed)
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
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Optional base random seed. If omitted, each episode uses OS entropy. "
            "If set, episode seed = seed + episode_index."
        ),
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
    return

    # example_episodes = 6
    # example_videos = []
    # for i in range(example_episodes):
    #     seed = _episode_seed(args.seed, i)
    #     images = generate_sequence(env, seed=seed)
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
            seed = _episode_seed(args.seed, i)
            images = generate_sequence(env, seed=seed)
            np.savez(os.path.join(train_dir, f"{i}.npz"), video=images)

        print()

        for i in range(TEST_EPISODES):
            print(f"\nGenerating test episode {i+1}/{TEST_EPISODES}")
            seed = _episode_seed(args.seed, TRAIN_EPISODES + i)
            images = generate_sequence(env, seed=seed)
            np.savez(os.path.join(test_dir, f"{i}.npz"), video=images)

        env.close()
    else:
        ctx = mp.get_context("spawn")
        train_jobs = [("train", i, train_dir, args.seed) for i in range(TRAIN_EPISODES)]
        test_jobs = [
            (
                "test",
                i,
                test_dir,
                None if args.seed is None else args.seed + TRAIN_EPISODES,
            )
            for i in range(TEST_EPISODES)
        ]
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
