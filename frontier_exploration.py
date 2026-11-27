import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from exploration_simulation import ExplorationEnv
from frontier_policy import FrontierExplorer


def evaluate(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    env = ExplorationEnv(
        grid_size=args.grid_size,
        sensor_radius=args.sensor_radius,
        max_steps=args.max_steps,
        target_coverage=args.target_coverage
    )

    rows = []

    try:
        for episode in range(args.episodes):
            seed = args.test_seed + episode
            obs, info = env.reset(seed=seed)
            agent = FrontierExplorer()

            done = info["coverage"] >= args.target_coverage

            while not done:
                action = agent.act(obs)

                if action is None:
                    break

                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated

            row = {
                "algorithm": "frontier",
                "seed": seed,
                "coverage": info["coverage"],
                "steps": info["steps"],
                "collisions": info["collisions"],
                "redundant_moves": info["redundant_moves"],
                "success": int(
                    info["coverage"] >= args.target_coverage
                ),
            }

            rows.append(row)

            print(
                f"Test map {seed}: "
                f"coverage={row['coverage']:.1%}, "
                f"steps={row['steps']}, "
                f"collisions={row['collisions']}, "
                f"redundant={row['redundant_moves']}, "
                f"success={row['success']}"
            )

            if episode == 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))

                env.render(show_true_map=True, ax=axes[0])
                env.render(show_true_map=False, ax=axes[1])

                fig.suptitle(
                    f"Frontier exploration, unseen map (seed {seed})"
                )

                fig.tight_layout()
                fig.savefig(
                    output / "evaluation_frontier.png",
                    dpi=150
                )

                plt.close(fig)

        csv_path = output / "evaluation_frontier.csv"

        with csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=list(rows[0])
            )

            writer.writeheader()
            writer.writerows(rows)

        print(f"\nEvaluation mean across {len(rows)} unseen maps:")

        for key in [
            "coverage",
            "steps",
            "collisions",
            "redundant_moves",
            "success"
        ]:
            mean = np.mean([row[key] for row in rows])
            print(f"  {key}: {mean:.3f}")

        print(f"Saved metrics: {csv_path}")

    finally:
        env.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Frontier-based robot exploration baseline"
    )

    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--test-seed", type=int, default=40000)
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--sensor-radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--target-coverage", type=float, default=0.95)
    parser.add_argument("--output-dir", default="models")

    args = parser.parse_args()

    if args.episodes < 1:
        parser.error("--episodes must be at least 1")

    return args


if __name__ == "__main__":
    evaluate(parse_args())
