import argparse
import csv
from collections import deque
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from exploration_simulation import ExplorationEnv
MOVES = ExplorationEnv.MOVES
def neighbors(cell, shape):
    r, c = cell
    for action, (dr, dc) in enumerate(MOVES):
        rr, cc = r + dr, c + dc
        if 0 <= rr < shape[0] and 0 <= cc < shape[1]:
            yield action, (rr, cc)
def shortest_paths(observation, start):
    queue = deque([start])
    distances = {start: 0}
    first_actions = {}
    while queue:
        current = queue.popleft()
        for action, nxt in neighbors(current, observation.shape):
            if nxt in distances or observation[nxt] not in (0, 2, 3):
                continue
            distances[nxt] = distances[current] + 1
            first_actions[nxt] = action if current == start else first_actions[current]
            queue.append(nxt)
    return distances, first_actions
class FrontierExplorer:
    def __init__(self):
        self.last_position = None
        self.stalled = {}
    def act(self, observation):
        positions = np.argwhere(observation == 3)
        if len(positions) != 1:
            raise ValueError('Observation must contain exactly one robot position')
        start = tuple(map(int, positions[0]))
        distances, first_actions = shortest_paths(observation, start)
        candidates = []
        for cell, distance in distances.items():
            unknown = sum(observation[nxt] == -1 for _, nxt in neighbors(cell, observation.shape))
            if unknown == 0:
                continue
            score = 3.0 * unknown - distance - 5.0 * self.stalled.get(cell, 0)
            if cell != start:
                candidates.append((score, -distance, cell, first_actions[cell]))
            else:
                self.stalled[cell] = self.stalled.get(cell, 0) + 1
        if candidates:
            return max(candidates)[3]
        options = [(observation[nxt] == 0, observation[nxt] == 2, action)
                   for action, nxt in neighbors(start, observation.shape)
                   if observation[nxt] in (0, 2, 3)]
        if options:
            return max(options)[2]
        return 0
def evaluate(args):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    env = ExplorationEnv(grid_size=args.grid_size,
                         sensor_radius=args.sensor_radius,
                         max_steps=args.max_steps,
                         target_coverage=args.target_coverage)
    rows = []
    try:
        for episode in range(args.episodes):
            seed = args.test_seed + episode
            obs, info = env.reset(seed=seed)
            agent = FrontierExplorer()
            done = False
            while not done:
                action = agent.act(obs)
                obs, _, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            row = {
                'algorithm': 'frontier', 'seed': seed,
                'coverage': info['coverage'], 'steps': info['steps'],
                'collisions': info['collisions'],
                'redundant_moves': info['redundant_moves'],
                'success': int(info['coverage'] >= args.target_coverage),
            }
            rows.append(row)
            print(f"Test map {seed}: coverage={row['coverage']:.1%}, "
                  f"steps={row['steps']}, collisions={row['collisions']}, "
                  f"redundant={row['redundant_moves']}, success={row['success']}")
            if episode == 0:
                fig, axes = plt.subplots(1, 2, figsize=(12, 5))
                env.render(show_true_map=True, ax=axes[0])
                env.render(show_true_map=False, ax=axes[1])
                fig.suptitle(f'Frontier exploration, unseen map (seed {seed})')
                fig.tight_layout()
                fig.savefig(output / 'evaluation_frontier.png', dpi=150)
                plt.close(fig)
        csv_path = output / 'evaluation_frontier.csv'
        with csv_path.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f'\nEvaluation mean across {len(rows)} unseen maps:')
        for key in ['coverage', 'steps', 'collisions', 'redundant_moves', 'success']:
            print(f'  {key}: {np.mean([row[key] for row in rows]):.3f}')
        print(f'Saved metrics: {csv_path}')
    finally:
        env.close()
def parse_args():
    parser = argparse.ArgumentParser(description='Frontier-based robot exploration baseline')
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--test-seed', type=int, default=10000)
    parser.add_argument('--grid-size', type=int, default=20)
    parser.add_argument('--sensor-radius', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=500)
    parser.add_argument('--target-coverage', type=float, default=0.95)
    parser.add_argument('--output-dir', default='models')
    args = parser.parse_args()
    if args.episodes < 1:
        parser.error('--episodes must be at least 1')
    return args
if __name__ == '__main__':
    evaluate(parse_args())
