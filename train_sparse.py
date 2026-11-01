import csv
from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from gym_texture_env import TextureGymEnv

ROOT = Path.home() / "pai_ws/texture_experiment"
OUT = ROOT / "results" / "sparse_final"
OUT.mkdir(parents=True, exist_ok=True)

torch.set_num_threads(2)


class EpisodeLogger(BaseCallback):
    def __init__(self):
        super().__init__()
        self.episode = 0

    def _on_step(self):
        for done, info in zip(
            self.locals["dones"],
            self.locals["infos"]
        ):
            if done:
                self.episode += 1
                ep = info.get("episode", {})

                row = [
                    "sparse",
                    42,
                    self.episode,
                    self.num_timesteps,
                    ep.get("l", ""),
                    ep.get("r", ""),
                    info.get("elapsed", ""),
                    info.get("success", False),
                ]

                with open(OUT / "episodes.csv", "a") as f:
                    csv.writer(f).writerow(row)

                print(
                    f"Episode {self.episode} | "
                    f"steps={ep.get('l')} | "
                    f"reward={ep.get('r')} | "
                    f"success={info.get('success')}",
                    flush=True
                )
        return True


with open(OUT / "episodes.csv", "w") as f:
    csv.writer(f).writerow([
        "condition", "seed", "episode",
        "total_steps", "episode_steps",
        "episode_reward", "sim_seconds", "success"
    ])

env = TextureGymEnv(max_steps=40)

try:
    model = PPO(
        "MultiInputPolicy",
        env,
        learning_rate=3e-4,
        n_steps=32,
        batch_size=16,
        n_epochs=4,
        gamma=0.99,
        policy_kwargs={
            "features_extractor_kwargs": {
                "cnn_output_dim": 64
            }
        },
        device="cpu",
        verbose=1,
        seed=42,
    )

    print("=== SPARSE TRAINING START ===", flush=True)

    model.learn(
        total_timesteps=128,
        callback=EpisodeLogger(),
    )

    model.save(str(OUT / "ppo_model"))

    print("=== SPARSE TRAINING COMPLETE ===")
    print("Log:", OUT / "episodes.csv")
    print("Model:", OUT / "ppo_model.zip")

finally:
    env.close()
