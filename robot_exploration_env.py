from collections import deque
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class ExplorationEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))

    def __init__(self, grid_size=20, sensor_radius=3, max_steps=500,
                 target_coverage=0.95, render_mode=None):
        super().__init__()
        if grid_size < 10:
            raise ValueError("grid_size must be >= 10")
        if sensor_radius < 1:
            raise ValueError("sensor_radius must be >= 1")
        if not (0 < target_coverage <= 1):
            raise ValueError("target_coverage must be in (0, 1]")
        self.grid_size = int(grid_size)
        self.sensor_radius = int(sensor_radius)
        self.max_steps = int(max_steps)
        self.target_coverage = float(target_coverage)
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=-1, high=3, shape=(grid_size, grid_size), dtype=np.int8
        )
        self.grid = None
        self.known = None
        self.visited = None
        self.position = None
        self.path = []
        self.steps = 0
        self.collisions = 0
        self.redundant_moves = 0
        self.reachable_free = 0

    @staticmethod
    def _flood(grid, start):
        seen = {tuple(start)}
        queue = deque([tuple(start)])
        height, width = grid.shape
        while queue:
            r, c = queue.popleft()
            for dr, dc in ExplorationEnv.MOVES:
                rr, cc = r + dr, c + dc
                if (0 <= rr < height and 0 <= cc < width
                    and grid[rr, cc] == 0 and (rr, cc) not in seen):
                    seen.add((rr, cc))
                    queue.append((rr, cc))
        return seen

    def _generate_map(self):
        """Create a connected floor plan from random rooms and corridors."""
        n = self.grid_size
        grid = np.ones((n, n), dtype=np.int8)  # 1=wall, 0=floor
        centers = []
        room_count = int(self.np_random.integers(5, 9))
        max_room = max(4, n // 3)
        for _ in range(room_count):
            h = int(self.np_random.integers(3, max_room + 1))
            w = int(self.np_random.integers(3, max_room + 1))
            r = int(self.np_random.integers(1, n - h))
            c = int(self.np_random.integers(1, n - w))
            grid[r:r + h, c:c + w] = 0
            centers.append((r + h // 2, c + w // 2))

        # Link room centers with L-shaped corridors, guaranteeing connectivity.
        for (r1, c1), (r2, c2) in zip(centers[:-1], centers[1:]):
            if self.np_random.integers(2) == 0:
                grid[min(r1, r2):max(r1, r2) + 1, c1] = 0
                grid[r2, min(c1, c2):max(c1, c2) + 1] = 0
            else:
                grid[r1, min(c1, c2):max(c1, c2) + 1] = 0
                grid[min(r1, r2):max(r1, r2) + 1, c2] = 0

        # Insert a few pillars only if they do not disconnect the floor plan.
        for _ in range(n // 2):
            floors = np.argwhere(grid[1:-1, 1:-1] == 0)
            if len(floors) < 2:
                break
            idx = int(self.np_random.integers(len(floors)))
            r, c = floors[idx] + 1
            grid[r, c] = 1
            remaining = np.argwhere(grid == 0)
            if len(self._flood(grid, remaining[0])) != len(remaining):
                grid[r, c] = 0
        return grid

    @staticmethod
    def _line_cells(a, b):
        """Integer grid line; wall cells block sight beyond themselves."""
        r0, c0 = a
        r1, c1 = b
        steps = max(abs(r1 - r0), abs(c1 - c0))
        if steps == 0:
            return [(r0, c0)]
        return [(round(r0 + (r1 - r0) * i / steps),
                 round(c0 + (c1 - c0) * i / steps)) for i in range(steps + 1)]

    def _sense(self):
        """Reveal a local square region without seeing through walls."""
        r0, c0 = self.position
        radius = self.sensor_radius
        n = self.grid_size
        for r in range(max(0, r0 - radius), min(n, r0 + radius + 1)):
            for c in range(max(0, c0 - radius), min(n, c0 + radius + 1)):
                for rr, cc in self._line_cells((r0, c0), (r, c)):
                    self.known[rr, cc] = self.grid[rr, cc]
                    if self.grid[rr, cc] == 1:
                        break

    def _observation(self):
        obs = self.known.copy()
        obs[self.visited] = 2
        obs[self.position] = 3
        return obs

    def _info(self, newly_discovered=0):
        explored = int(np.count_nonzero((self.known == 0) & (self.grid == 0)))
        return {
            "coverage": explored / self.reachable_free,
            "explored_free_cells": explored,
            "reachable_free_cells": self.reachable_free,
            "new_free_cells": newly_discovered,
            "steps": self.steps,
            "collisions": self.collisions,
            "redundant_moves": self.redundant_moves,
        }

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.grid = self._generate_map()
        floors = np.argwhere(self.grid == 0)
        self.reachable_free = len(floors)
        self.position = tuple(map(int, floors[self.np_random.integers(len(floors))]))
        self.known = np.full(self.grid.shape, -1, dtype=np.int8)
        self.visited = np.zeros(self.grid.shape, dtype=bool)
        self.visited[self.position] = True
        self.steps = self.collisions = self.redundant_moves = 0
        self.path = [self.position]
        self._sense()
        return self._observation(), self._info()

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action {action}; expected 0,1,2,3")
        self.steps += 1
        dr, dc = self.MOVES[int(action)]
        r, c = self.position
        nr, nc = r + dr, c + dc
        old_known_free = int(np.count_nonzero(self.known == 0))
        reward = -0.01  # discourage slow exploration
        if not (0 <= nr < self.grid_size and 0 <= nc < self.grid_size) or self.grid[nr, nc] == 1:
            self.collisions += 1
            reward -= 0.20
        else:
            self.position = (nr, nc)
            if self.visited[self.position]:
                self.redundant_moves += 1
                reward -= 0.03
            self.visited[self.position] = True
            self._sense()
        self.path.append(self.position)
        new_free = int(np.count_nonzero(self.known == 0)) - old_known_free
        reward += 0.10 * new_free
        info = self._info(new_free)
        terminated = info["coverage"] >= self.target_coverage
        truncated = self.steps >= self.max_steps and not terminated
        if terminated:
            reward += 5.0
        return self._observation(), float(reward), terminated, truncated, info

    def render(self, show_true_map=False, ax=None):
        """Returns (fig, ax). For notebooks, call plt.show() afterward."""
        import matplotlib.pyplot as plt
        from matplotlib.colors import BoundaryNorm, ListedColormap
        if self.grid is None:
            raise RuntimeError("Call reset() before render().")
        values = self.grid.copy() if show_true_map else self._observation()
        if show_true_map:
            # Convert ground truth into same display labels.
            values = np.where(values == 1, 1, 0).astype(np.int8)
            values[self.position] = 3
        # -1 unknown=dark; 0 free=cream; 1 wall=charcoal;
        # 2 visited=light blue; 3 robot=orange
        palette = ListedColormap(["#89939d", "#f4f1e8", "#263140", "#90c7dc", "#f0833c"])
        norm = BoundaryNorm([-1.5, -0.5, 0.5, 1.5, 2.5, 3.5], palette.N)
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        else:
            fig = ax.figure
        ax.imshow(values, cmap=palette, norm=norm, interpolation="none")
        if len(self.path) > 1:
            points = np.asarray(self.path)
            ax.plot(points[:, 1], points[:, 0], linewidth=1, color="#c94747", alpha=0.8)
        ax.set_title(("True map" if show_true_map else "Robot's discovered map")
                     + f" | coverage {self._info()['coverage']:.1%}")
        ax.set_xticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, self.grid_size, 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.25)
        ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
        return fig, ax
