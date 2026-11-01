from pathlib import Path

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback

from gym_texture_env import TextureGymEnv

ROOT = Path.home() / "pai_ws/texture_experiment"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

torch.set_num_threads(2)


class ProgressCallback(BaseCallback):
    def _on_step(self):
        print(f"PPO step: {self.num_timesteps}", flush=True)
        return True


env = TextureGymEnv(max_steps=30)

try:
    model = PPO(
        policy="MultiInputPolicy",
        env=env,
        learning_rate=3e-4,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
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

    print("=== PPO TRAINING START ===", flush=True)

    model.learn(
        total_timesteps=8,
        callback=ProgressCallback(),
    )

    path = MODEL_DIR / "ppo_texture_smoke"
    model.save(str(path))

    print("=== PPO TRAINING COMPLETE ===")
    print("Model saved:", str(path) + ".zip")

finally:
    env.close()
