"""Reproduce the PPO diagnosis, training, and three-way evaluation.

Run this script with diagnose, train, and evaluate in that order. The original
checkpoints remain in models/ and the new artifacts go to models/experiment/.
"""

import argparse
import csv
import json
from collections import Counter, deque
from pathlib import Path
from types import SimpleNamespace

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

from curiosity_ppo import make_env
from exploration_simulation import ExplorationEnv

plt.switch_backend("Agg")

TRAIN_MAPS = tuple(range(20000, 20512))
TUNING_MAPS = tuple(range(30000, 30010))
TEST_MAPS = tuple(range(40000, 40020))
TRAIN_SEEDS = (42, 43, 44)
ALGORITHMS = ("standard", "curiosity")
ENVIRONMENT = dict(grid_size=20, sensor_radius=3, max_steps=500,
                   target_coverage=0.95, curiosity_weight=0.05, device="cpu")
MOVES = ExplorationEnv.MOVES
ACTION_NAMES = ("up", "down", "left", "right")


class TrainingMaps(gym.Wrapper):
    """Sample only from the training layouts while respecting PPO's RNG seed."""

    def reset(self, *, seed=None, options=None):
        if seed is not None or not hasattr(self, "map_rng"):
            self.map_rng = np.random.default_rng(seed)
        map_seed = int(self.map_rng.choice(TRAIN_MAPS))
        return self.env.reset(seed=map_seed, options=options)


class LocalMapFeatures(BaseFeaturesExtractor):
    """Read a robot-centered crop so the policy can transfer between positions."""

    def __init__(self, observation_space, radius=4):
        width = 2 * radius + 1
        super().__init__(observation_space, features_dim=5 * width * width)
        self.radius = radius
        self.register_buffer("offsets", torch.arange(-radius, radius + 1))

    def forward(self, observations):
        batch, _, _, width = observations.shape
        positions = observations[:, 4].flatten(1).argmax(dim=1)
        rows = positions // width + self.radius
        cols = positions % width + self.radius
        padded = torch.nn.functional.pad(
            observations, (self.radius,) * 4,
        )
        local = padded.permute(0, 2, 3, 1)[
            torch.arange(batch, device=observations.device)[:, None, None],
            rows[:, None, None] + self.offsets[None, :, None],
            cols[:, None, None] + self.offsets[None, None, :],
        ]
        return local.reshape(batch, -1)


class KnownWallPolicy(ActorCriticPolicy):
    """Use the same known-wall mask in rollouts, PPO updates, and prediction."""

    def _distribution(self, obs, actor_latent):
        batch, _, _, width = obs.shape
        position = obs[:, 4].flatten(1).argmax(dim=1)
        moves = position.new_tensor(MOVES)
        rows = position[:, None] // width + moves[None, :, 0]
        cols = position[:, None] % width + moves[None, :, 1]
        walls = obs[torch.arange(batch, device=obs.device)[:, None], 2, rows, cols]
        logits = self.action_net(actor_latent).masked_fill(walls.bool(), -torch.inf)
        return self.action_dist.proba_distribution(action_logits=logits)

    def forward(self, obs, deterministic=False):
        actor, critic = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, actor)
        actions = distribution.get_actions(deterministic=deterministic)
        return actions, self.value_net(critic), distribution.log_prob(actions)

    def evaluate_actions(self, obs, actions):
        actor, critic = self.mlp_extractor(self.extract_features(obs))
        distribution = self._distribution(obs, actor)
        return self.value_net(critic), distribution.log_prob(actions), distribution.entropy()

    def get_distribution(self, obs):
        actor = self.mlp_extractor.forward_actor(self.extract_features(obs))
        return self._distribution(obs, actor)


def frontier_action(observation):
    """Return the first action on a shortest known path to a frontier cell."""
    start = tuple(np.argwhere(observation == 3)[0])
    height, width = observation.shape
    queue = deque([(start, None)])
    visited = {start}
    while queue:
        (row, col), first = queue.popleft()
        for action, (dr, dc) in enumerate(MOVES):
            neighbor = row + dr, col + dc
            r, c = neighbor
            if not (0 <= r < height and 0 <= c < width):
                continue
            cell = observation[neighbor]
            if cell == -1:
                return action if first is None else first
            if cell in (0, 2) and neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, action if first is None else first))
    return None


def rollout(env, choose_action, map_seed):
    observation, info = env.reset(seed=map_seed)
    counts = Counter()
    collision_actions = Counter()
    target_reached = info["coverage"] >= ENVIRONMENT["target_coverage"]
    outcome = "target_reached" if target_reached else ""
    while not outcome:
        action = choose_action(observation)
        if action is None:
            outcome = "no_frontier"
            break
        counts[ACTION_NAMES[action]] += 1
        collisions = info["collisions"]
        observation, _, terminated, truncated, info = env.step(action)
        if info["collisions"] > collisions:
            collision_actions[ACTION_NAMES[action]] += 1
        if terminated:
            outcome = "target_reached"
        elif truncated:
            outcome = "time_limit"
    success = outcome == "target_reached"
    return {
        "map_seed": map_seed,
        "coverage": info["coverage"],
        "steps": info["steps"],
        "steps_to_95_percent": info["steps"] if success else None,
        "success": int(success),
        "outcome": outcome,
        "revisits": info["redundant_moves"],
        "collisions": info["collisions"],
        "unique_positions": len(set(env.unwrapped.path)),
        **{f"action_{name}": counts[name] for name in ACTION_NAMES},
        **{f"collision_{name}": collision_actions[name] for name in ACTION_NAMES},
    }


def save_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_trajectory(env, path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    env.unwrapped.render(show_true_map=True, ax=axes[0])
    env.unwrapped.render(ax=axes[1])
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def policy_results(checkpoint, maps, deterministic, trajectory=None):
    model = PPO.load(checkpoint, device="cpu")
    env = make_env(SimpleNamespace(**ENVIRONMENT))
    rows = []
    try:
        for index, seed in enumerate(maps):
            torch.manual_seed(seed)
            rows.append(rollout(env, lambda obs: int(model.predict(
                obs, deterministic=deterministic)[0]), seed))
            if index == 0 and trajectory is not None:
                save_trajectory(env, trajectory)
    finally:
        env.close()
    return rows


def diagnose(output, original):
    rows = []
    for algorithm in ALGORITHMS:
        checkpoint = original / f"ppo_{algorithm}.zip"
        for mode in ("deterministic", "sampled"):
            results = policy_results(checkpoint, TUNING_MAPS, mode == "deterministic",
                                     output / f"original_{algorithm}_{mode}.png")
            rows.extend(dict(algorithm=algorithm, mode=mode, **row) for row in results)
    save_csv(output / "original_diagnosis.csv", rows)


def train(output, timesteps, train_seeds):
    import random
    from importlib.metadata import version

    settings = dict(n_steps=512, batch_size=64, n_epochs=5, learning_rate=3e-4,
                    gamma=0.99, ent_coef=0.01)
    metadata = dict(environment=ENVIRONMENT, ppo=settings,
                    training_seeds=train_seeds, timesteps=timesteps,
                    actual_timesteps=((timesteps + 511) // 512) * 512,
                    training_map_seeds=TRAIN_MAPS, tuning_map_seeds=TUNING_MAPS,
                    test_map_seeds=TEST_MAPS, local_crop_radius=4,
                    policy_layers=[128, 128], known_wall_mask=True,
                    libraries={name: version(name) for name in
                               ("stable-baselines3", "gymnasium", "torch", "numpy", "matplotlib")})
    (output / "settings.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    for seed in train_seeds:
        for algorithm in ALGORITHMS:
            print(f"Training {algorithm} PPO, seed {seed}", flush=True)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)
            env = make_env(SimpleNamespace(**ENVIRONMENT), curiosity=algorithm == "curiosity")
            training_env = TrainingMaps(env)
            try:
                model = PPO(KnownWallPolicy, training_env, seed=seed, device="cpu",
                            policy_kwargs=dict(features_extractor_class=LocalMapFeatures,
                                               net_arch=[128, 128]), **settings)
                model.learn(total_timesteps=timesteps)
                model.save(output / f"ppo_{algorithm}_{seed}.zip")
                if algorithm == "curiosity":
                    env.save_icm(output / f"icm_{seed}.pt")
            finally:
                training_env.close()


def evaluate(output, train_seeds):
    tuning_rows = []
    by_mode = {}
    for mode in ("deterministic", "sampled"):
        rows = []
        for algorithm in ALGORITHMS:
            for seed in train_seeds:
                print(f"Tuning {algorithm} PPO, seed {seed}, {mode}", flush=True)
                checkpoint = output / f"ppo_{algorithm}_{seed}.zip"
                results = policy_results(checkpoint, TUNING_MAPS, mode == "deterministic")
                rows.extend(dict(algorithm=algorithm, train_seed=seed, mode=mode, **row)
                            for row in results)
        by_mode[mode] = rows
        tuning_rows.extend(rows)
    selected_mode = max(by_mode, key=lambda mode: (
        np.mean([row["success"] for row in by_mode[mode]]),
        np.mean([row["coverage"] for row in by_mode[mode]]),
    ))
    save_csv(output / "tuning_results.csv", tuning_rows)
    (output / "selected_mode.json").write_text(json.dumps({
        "mode": selected_mode,
        "selection": "Highest pooled tuning success rate, then mean coverage",
    }, indent=2), encoding="utf-8")

    final_rows = []
    baseline = ExplorationEnv(**{key: ENVIRONMENT[key] for key in
                                 ("grid_size", "sensor_radius", "max_steps", "target_coverage")})
    try:
        for map_seed in TEST_MAPS:
            row = rollout(baseline, frontier_action, map_seed)
            final_rows.append(dict(algorithm="frontier", train_seed=None, mode="bfs", **row))
            if map_seed == TEST_MAPS[0]:
                save_trajectory(baseline, output / "final_frontier.png")
    finally:
        baseline.close()

    for algorithm in ALGORITHMS:
        for seed in train_seeds:
            print(f"Testing {algorithm} PPO, seed {seed}, {selected_mode}", flush=True)
            checkpoint = output / f"ppo_{algorithm}_{seed}.zip"
            results = policy_results(checkpoint, TEST_MAPS, selected_mode == "deterministic",
                                     output / f"final_{algorithm}_{seed}.png")
            final_rows.extend(dict(algorithm=algorithm, train_seed=seed, mode=selected_mode, **row)
                              for row in results)
    save_csv(output / "final_results.csv", final_rows)

    summary = []
    for algorithm in ("frontier", *ALGORITHMS):
        rows = [row for row in final_rows if row["algorithm"] == algorithm]
        successful_steps = [row["steps_to_95_percent"] for row in rows if row["success"]]
        summary.append(dict(
            algorithm=algorithm, runs=len(rows),
            mean_final_coverage=np.mean([row["coverage"] for row in rows]),
            success_rate=np.mean([row["success"] for row in rows]),
            mean_steps_to_95_percent=np.mean(successful_steps) if successful_steps else None,
            time_limit_failures=sum(row["outcome"] == "time_limit" for row in rows),
            no_frontier_failures=sum(row["outcome"] == "no_frontier" for row in rows),
            mean_revisits=np.mean([row["revisits"] for row in rows]),
            mean_collisions=np.mean([row["collisions"] for row in rows]),
        ))
    save_csv(output / "final_summary.csv", summary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("diagnose", "train", "evaluate"))
    parser.add_argument("--output", type=Path, default=Path("models/experiment"))
    parser.add_argument("--original", type=Path, default=Path("models"))
    parser.add_argument("--timesteps", type=int, default=30720)
    parser.add_argument("--train-seeds", type=int, nargs="+", default=TRAIN_SEEDS)
    args = parser.parse_args()
    if args.timesteps < 512:
        parser.error("--timesteps must be at least 512")
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    if args.stage == "diagnose":
        diagnose(args.output, args.original)
    elif args.stage == "train":
        train(args.output, args.timesteps, args.train_seeds)
    else:
        evaluate(args.output, args.train_seeds)


if __name__ == "__main__":
    main()
