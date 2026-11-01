from pathlib import Path
import csv
import numpy as np
from stable_baselines3 import PPO
from gym_texture_env import TextureGymEnv

ROOT = Path.home() / "pai_ws/texture_experiment"
OUT = ROOT / "results/none_final/inference"
OUT.mkdir(parents=True, exist_ok=True)

model = PPO.load(
    str(ROOT / "results/none_final/ppo_model.zip"),
    device="cpu"
)

env = TextureGymEnv(max_steps=40)
records = []
total_reward = 0.0

try:
    print("Resetting environment...", flush=True)
    obs, _ = env.reset(seed=2026)

    print("=== INFERENCE START ===", flush=True)

    for step in range(1, 41):
        action, _ = model.predict(
            obs, deterministic=True
        )

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)

        records.append({
            "step": step,
            "action": np.array2string(
                np.asarray(action), precision=4
            ),
            "reward": float(reward),
            "distance_m": info.get("distance_m", ""),
            "success": info.get("success", False),
            "terminated": terminated,
            "truncated": truncated
        })

        print(
            f"Step {step}: "
            f"reward={float(reward):.6f}, "
            f"distance={info.get('distance_m')}, "
            f"success={info.get('success', False)}",
            flush=True
        )

        if terminated or truncated:
            break

    with open(OUT / "inference.csv", "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=records[0].keys()
        )
        writer.writeheader()
        writer.writerows(records)

    print("=== INFERENCE COMPLETE ===")
    print("Steps:", len(records))
    print("Total reward:", round(total_reward, 6))
    print("Success:", info.get("success", False))
    print("Log:", OUT / "inference.csv")

finally:
    env.close()
