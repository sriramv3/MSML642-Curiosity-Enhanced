import argparse
import csv
import random
from collections import deque
from pathlib import Path

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor

from exploration_simulation import ExplorationEnv


# Known map values -1,0,1,2,3 are encoded as 5 one-hot channels.
# No ground-truth map is ever passed to PPO or ICM.
class OneHotMapObservation(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0.0, high=1.0, shape=(5, h, w), dtype=np.float32
        )

    def observation(self, observation):
        indices = observation.astype(np.int64) + 1
        # 5 channels: unknown, free, wall, visited, current position
        return np.eye(5, dtype=np.float32)[indices].transpose(2, 0, 1)


class IntrinsicCuriosityModule(nn.Module):
    def __init__(self, observation_shape, n_actions=4, feature_size=128):
        super().__init__()
        input_size = int(np.prod(observation_shape))
        self.n_actions = n_actions
        self.encoder = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size, 256), nn.ReLU(),
            nn.Linear(256, feature_size), nn.ReLU(),
        )
        self.inverse_model = nn.Sequential(
            nn.Linear(2 * feature_size, 256), nn.ReLU(),
            nn.Linear(256, n_actions),
        )
        self.forward_model = nn.Sequential(
            nn.Linear(feature_size + n_actions, 256), nn.ReLU(),
            nn.Linear(256, feature_size),
        )

    def forward(self, obs, actions, next_obs):
        feature = self.encoder(obs)
        next_feature = self.encoder(next_obs)
        predicted_action = self.inverse_model(torch.cat([feature, next_feature], dim=1))
        one_hot_action = F.one_hot(actions.long(), num_classes=self.n_actions).float()
        predicted_next_feature = self.forward_model(
            torch.cat([feature, one_hot_action], dim=1)
        )
        return predicted_action, predicted_next_feature, next_feature

    @torch.no_grad()
    def prediction_errors(self, obs, actions, next_obs):
        # Reward computed under the current network, before parameter updates.
        self.eval()
        _, predicted, target = self.forward(obs, actions, next_obs)
        return (predicted - target).square().mean(dim=1)


class CuriosityRewardWrapper(gym.Wrapper):
    """Adds intrinsic reward online and trains ICM from a replay minibatch.

    PPO is trained with its usual Stable-Baselines3 `.learn()` loop. Each call
    to step() computes an intrinsic bonus and updates ICM periodically.
    """

    def __init__(self, env, curiosity_weight=0.05, learning_rate=1e-3,
                 batch_size=64, replay_capacity=10000, update_every=32,
                 warmup=128, device="cpu"):
        super().__init__(env)
        self.device = torch.device(device)
        self.icm = IntrinsicCuriosityModule(env.observation_space.shape,
                                            env.action_space.n).to(self.device)
        self.optimizer = torch.optim.Adam(self.icm.parameters(), lr=learning_rate)
        self.weight = float(curiosity_weight)
        self.batch_size = int(batch_size)
        self.update_every = int(update_every)
        self.warmup = int(warmup)
        self.replay = deque(maxlen=replay_capacity)
        self.obs = None
        self.total_steps = 0
        self.training_enabled = True
        self.last_icm_loss = None
        # Running scale prevents an arbitrarily large early bonus dominating PPO.
        self.mean_error = 0.0
        self.error_updates = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.obs = obs.copy()
        return obs, info

    def step(self, action):
        next_obs, extrinsic, terminated, truncated, info = self.env.step(action)
        state_tensor = torch.as_tensor(self.obs[None], dtype=torch.float32, device=self.device)
        next_tensor = torch.as_tensor(next_obs[None], dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor([int(action)], dtype=torch.long, device=self.device)
        error = float(self.icm.prediction_errors(state_tensor, action_tensor, next_tensor).item())

        # Exponential moving average with warmup bias correction for reward scale.
        self.error_updates += 1
        self.mean_error = 0.99 * self.mean_error + 0.01 * error
        corrected_scale = self.mean_error / (1.0 - 0.99 ** self.error_updates)
        normalized_error = min(error / max(corrected_scale, 1e-6), 5.0)
        intrinsic = self.weight * normalized_error

        if self.training_enabled:
            self.replay.append((self.obs.copy(), int(action), next_obs.copy()))
            self.total_steps += 1
            if len(self.replay) >= max(self.warmup, self.batch_size) and \
                    self.total_steps % self.update_every == 0:
                self._update_icm()
        self.obs = next_obs.copy()
        info = dict(info)
        info["extrinsic_reward"] = float(extrinsic)
        info["intrinsic_reward"] = float(intrinsic)
        if self.last_icm_loss is not None:
            info["icm_loss"] = self.last_icm_loss
        return next_obs, float(extrinsic + intrinsic), terminated, truncated, info

    def _update_icm(self):
        indices = np.random.choice(len(self.replay), size=self.batch_size, replace=False)
        samples = [self.replay[int(i)] for i in indices]
        obs = torch.as_tensor(np.stack([s[0] for s in samples]),
                              dtype=torch.float32, device=self.device)
        actions = torch.as_tensor([s[1] for s in samples],
                                  dtype=torch.long, device=self.device)
        next_obs = torch.as_tensor(np.stack([s[2] for s in samples]),
                                   dtype=torch.float32, device=self.device)
        self.icm.train()
        action_logits, prediction, next_feature = self.icm(obs, actions, next_obs)
        inverse_loss = F.cross_entropy(action_logits, actions)
        forward_loss = F.mse_loss(prediction, next_feature.detach())
        loss = inverse_loss + 0.2 * forward_loss
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.icm.parameters(), 1.0)
        self.optimizer.step()
        self.last_icm_loss = float(loss.detach().item())

    def save_icm(self, filepath):
        torch.save({"model": self.icm.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "mean_error": self.mean_error,
                    "error_updates": self.error_updates,
                    "total_steps": self.total_steps}, filepath)


class ExplorationMetricsCallback(BaseCallback):
    def __init__(self, log_every=5000):
        super().__init__()
        self.log_every = log_every
        self.finished = 0

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                self.finished += 1
                if self.finished <= 3 or self.finished % 20 == 0:
                    print(f"Episode {self.finished:4d} | "
                          f"coverage={info.get('coverage', 0):6.1%} | "
                          f"steps={info.get('steps', 0):4d} | "
                          f"collisions={info.get('collisions', 0):4d}")
            if "icm_loss" in info:
                self.logger.record("curiosity/icm_loss", info["icm_loss"])
                self.logger.record("curiosity/bonus", info["intrinsic_reward"])
        return True


def make_env(args, curiosity=False):
    env = ExplorationEnv(grid_size=args.grid_size,
                         sensor_radius=args.sensor_radius,
                         max_steps=args.max_steps,
                         target_coverage=args.target_coverage)
    env = OneHotMapObservation(env)
    if curiosity:
        env = CuriosityRewardWrapper(env, curiosity_weight=args.curiosity_weight,
                                     device=args.device)
    return env


def model_path(args):
    return Path(args.output_dir) / f"ppo_{args.algorithm}.zip"


def train(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    is_curiosity = args.algorithm == "curiosity"
    env = make_env(args, curiosity=is_curiosity)
    check_env(env, warn=True)
    env.action_space.seed(args.seed)
    env.reset(seed=args.seed)
    monitored_env = Monitor(env)
    model = PPO("MlpPolicy", monitored_env,
                learning_rate=3e-4, n_steps=1024, batch_size=64,
                n_epochs=5, gamma=0.99, ent_coef=0.01,
                policy_kwargs=dict(net_arch=[128, 128]),
                seed=args.seed, verbose=1, device=args.device)
    print(f"Training {args.algorithm} PPO for {args.timesteps:,} steps")
    try:
        model.learn(total_timesteps=args.timesteps,
                    callback=ExplorationMetricsCallback())
        model.save(str(model_path(args)))
        if is_curiosity:
            env.save_icm(out / "curiosity_icm.pt")
        print(f"Saved PPO model: {model_path(args)}")
        if is_curiosity:
            print(f"Saved ICM model: {out / 'curiosity_icm.pt'}")
    finally:
        monitored_env.close()


def evaluate(args):
    path = model_path(args)
    if not path.is_file():
        raise FileNotFoundError(f"No trained model at {path}. Run --mode train first.")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    model = PPO.load(str(path), device=args.device)
    # Pure environment: no curiosity rewards during evaluation.
    env = make_env(args, curiosity=False)
    rows = []
    for episode in range(args.episodes):
        test_seed = args.test_seed + episode
        obs, info = env.reset(seed=test_seed)
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(int(action))
            done = terminated or truncated
        row = {
            "algorithm": args.algorithm,
            "seed": test_seed,
            "coverage": info["coverage"],
            "steps": info["steps"],
            "collisions": info["collisions"],
            "redundant_moves": info["redundant_moves"],
            "success": int(info["coverage"] >= args.target_coverage),
        }
        rows.append(row)
        print(f"Test map {test_seed}: coverage={row['coverage']:.1%}, "
              f"steps={row['steps']}, collisions={row['collisions']}, "
              f"redundant={row['redundant_moves']}, success={row['success']}")
        if episode == 0:
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            env.unwrapped.render(show_true_map=True, ax=axes[0])
            env.unwrapped.render(show_true_map=False, ax=axes[1])
            fig.suptitle(f"{args.algorithm.title()} PPO, unseen map (seed {test_seed})")
            fig.tight_layout()
            image_path = output / f"evaluation_{args.algorithm}.png"
            fig.savefig(image_path, dpi=150)
            plt.close(fig)
            print(f"Saved exploration image: {image_path}")

    csv_path = output / f"evaluation_{args.algorithm}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("\nEvaluation mean across", len(rows), "unseen maps:")
    for key in ["coverage", "steps", "collisions", "redundant_moves", "success"]:
        print(f"  {key}: {np.mean([r[key] for r in rows]):.3f}")
    print(f"Saved metrics: {csv_path}")
    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate PPO + ICM exploration")
    parser.add_argument("--mode", choices=["train", "evaluate"], default="train")
    parser.add_argument("--algorithm", choices=["curiosity", "standard"], default="curiosity")
    parser.add_argument("--timesteps", type=int, default=100000)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42,
                        help="training random seed (maps still change across episodes)")
    parser.add_argument("--test-seed", type=int, default=10000,
                        help="first held-out test map seed")
    parser.add_argument("--grid-size", type=int, default=20)
    parser.add_argument("--sensor-radius", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--target-coverage", type=float, default=0.95)
    parser.add_argument("--curiosity-weight", type=float, default=0.05)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-dir", default="models")
    return parser.parse_args()


if __name__ == "__main__":
    config = parse_args()
    if config.mode == "train":
        train(config)
    else:
        evaluate(config)
