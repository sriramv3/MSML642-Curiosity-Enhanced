from collections import deque
import numpy as np

MOVES = ((-1, 0), (1, 0), (0, -1), (0, 1))
FREE = (0, 2, 3)


def neighbors(position, shape):
    r, c = position
    for action, (dr, dc) in enumerate(MOVES):
        nr, nc = r + dr, c + dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1]:
            yield action, (nr, nc)


def find_frontiers(observation):
    obs = np.asarray(observation)
    if obs.ndim != 2:
        raise ValueError('Expected a 2D occupancy grid')
    return {tuple(map(int, cell)) for cell in np.argwhere(np.isin(obs, FREE))
            if any(obs[nxt] == -1 for _, nxt in neighbors(tuple(cell), obs.shape))}


def bfs_action(observation):
    obs = np.asarray(observation)
    positions = np.argwhere(obs == 3)
    if len(positions) != 1:
        raise ValueError('Observation must contain exactly one robot location (3)')
    start = tuple(map(int, positions[0]))
    frontiers = find_frontiers(obs)
    queue = deque([(start, None)])
    visited = {start}
    while queue:
        current, first_action = queue.popleft()
        if current != start and current in frontiers:
            return first_action
        for action, nxt in neighbors(current, obs.shape):
            if nxt not in visited and obs[nxt] in FREE:
                visited.add(nxt)
                queue.append((nxt, action if first_action is None else first_action))
    return None


class FrontierExplorer:
    def act(self, observation):
        return bfs_action(observation)
