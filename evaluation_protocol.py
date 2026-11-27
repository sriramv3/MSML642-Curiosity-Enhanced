from dataclasses import dataclass, asdict
import json
from pathlib import Path

TRAIN_MAPS = tuple(range(20000, 20512))
TUNING_MAPS = tuple(range(30000, 30010))
TEST_MAPS = tuple(range(40000, 40020))
TRAIN_SEEDS = (42, 43, 44)

@dataclass(frozen=True)
class EvaluationConfig:
    grid_size: int = 20
    sensor_radius: int = 3
    max_steps: int = 500
    target_coverage: float = 0.95
    ppo_timesteps: int = 30720
    ppo_rollout_steps: int = 512

    def environment_kwargs(self):
        return {key: getattr(self, key) for key in
                ('grid_size', 'sensor_radius', 'max_steps', 'target_coverage')}


def validate_protocol(train_maps=TRAIN_MAPS, tuning_maps=TUNING_MAPS,
                      test_maps=TEST_MAPS, training_seeds=TRAIN_SEEDS,
                      config=EvaluationConfig()):
    sets = [set(s) for s in (train_maps, tuning_maps, test_maps)]
    if any(not s for s in sets) or any(len(x) != len(set(x)) for x in
                                       (train_maps, tuning_maps, test_maps)):
        raise ValueError('Map splits must be nonempty and contain unique seeds')
    if sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2]:
        raise ValueError('Training, tuning, and test map sets must be disjoint')
    if len(set(training_seeds)) != len(training_seeds) or len(training_seeds) < 2:
        raise ValueError('Use multiple unique training seeds')
    if config.ppo_timesteps < config.ppo_rollout_steps:
        raise ValueError('PPO budget must be at least one rollout')
    if config.ppo_timesteps % config.ppo_rollout_steps:
        raise ValueError('PPO training steps must divide exactly into rollout steps')
    if not (0 < config.target_coverage <= 1) or config.max_steps < 1:
        raise ValueError('Invalid coverage target or step limit')
    return True


def manifest(config=EvaluationConfig(), training_seeds=TRAIN_SEEDS):
    validate_protocol(training_seeds=training_seeds, config=config)
    return {
        'environment': config.environment_kwargs(),
        'training_map_seeds': list(TRAIN_MAPS),
        'tuning_map_seeds': list(TUNING_MAPS),
        'final_test_map_seeds': list(TEST_MAPS),
        'training_seeds': list(training_seeds),
        'ppo_timesteps_per_variant_per_seed': config.ppo_timesteps,
        'ppo_rollout_steps': config.ppo_rollout_steps,
        'methods': ['frontier', 'standard', 'curiosity'],
        'tuning_rule': 'Choose deterministic versus sampled action mode using only tuning maps; hold mode fixed on final test maps.',
        'test_pairing': 'Same map seed reproduces the same map and starting position for every method.',
    }


def save_manifest(path, config=EvaluationConfig(), training_seeds=TRAIN_SEEDS):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest(config, training_seeds), indent=2) + '\n',
                    encoding='utf-8')
