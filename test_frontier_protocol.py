import unittest
import numpy as np
from frontier_policy import bfs_action, find_frontiers, MOVES
from evaluation_protocol import (TRAIN_MAPS, TUNING_MAPS, TEST_MAPS,
                                 TRAIN_SEEDS, EvaluationConfig,
                                 validate_protocol, manifest)

class TestFrontier(unittest.TestCase):
    def test_detect_frontier(self):
        obs = np.array([[1, 1, 1, 1], [1, 3, 0, -1],
                        [1, 1, 1, 1]], dtype=np.int8)
        self.assertEqual(find_frontiers(obs), {(1, 2)})
        self.assertEqual(bfs_action(obs), 3)

    def test_bfs_routes_around_wall(self):
        obs = np.array([[1, 1, 1, 1, 1],
                        [1, 3, 1, 0, -1],
                        [1, 0, 0, 0, 1],
                        [1, 1, 1, 1, 1]], dtype=np.int8)
        self.assertEqual(bfs_action(obs), 1)

    def test_no_frontier_returns_none(self):
        obs = np.array([[1, 1, 1], [1, 3, 1], [1, 1, -1]], dtype=np.int8)
        self.assertIsNone(bfs_action(obs))

    def test_never_steps_onto_unknown_or_wall(self):
        rng = np.random.default_rng(21)
        for _ in range(200):
            obs = rng.choice([-1, 0, 1, 2], (10, 10)).astype(np.int8)
            obs[5, 5] = 3
            action = bfs_action(obs)
            if action is not None:
                dr, dc = MOVES[action]
                self.assertIn(obs[5 + dr, 5 + dc], (0, 2, 3))

    def test_bad_robot_count(self):
        with self.assertRaises(ValueError):
            bfs_action(np.zeros((3, 3), dtype=np.int8))

class TestProtocol(unittest.TestCase):
    def test_disjoint_reproducible_splits(self):
        self.assertTrue(validate_protocol())
        self.assertFalse(set(TRAIN_MAPS) & set(TUNING_MAPS))
        self.assertFalse(set(TUNING_MAPS) & set(TEST_MAPS))
        self.assertEqual(len(TRAIN_SEEDS), 3)

    def test_overlapping_splits_rejected(self):
        with self.assertRaises(ValueError):
            validate_protocol((1, 2), (2, 3), (4, 5))

    def test_equal_budget_manifest(self):
        m = manifest()
        self.assertEqual(m['ppo_timesteps_per_variant_per_seed'], 30720)
        self.assertEqual(m['environment']['target_coverage'], 0.95)

class TestIntegration(unittest.TestCase):
    @unittest.skipUnless(__import__('importlib').util.find_spec('gymnasium'),
                         'gymnasium is not installed')
    def test_real_environment_coverage_and_step_limit(self):
        from exploration_simulation import ExplorationEnv
        for seed in (40000, 40001, 40002):
            env = ExplorationEnv(grid_size=20, sensor_radius=3,
                                 max_steps=500, target_coverage=0.95)
            try:
                observation, info = env.reset(seed=seed)
                first_position = env.position

                grid = env.grid.copy()
                env.reset(seed=seed)
                self.assertTrue(np.array_equal(grid, env.grid))
                self.assertEqual(first_position, env.position)
                while info['coverage'] < env.target_coverage and info['steps'] < env.max_steps:
                    action = bfs_action(observation)
                    if action is None:
                        break
                    observation, _, terminated, truncated, info = env.step(action)
                    if terminated or truncated:
                        break
                self.assertEqual(info['collisions'], 0)
                self.assertLessEqual(info['steps'], env.max_steps)
                self.assertTrue(info['coverage'] >= env.target_coverage or
                                info['steps'] == env.max_steps or
                                bfs_action(observation) is None)
            finally:
                env.close()

if __name__ == '__main__':
    unittest.main()
