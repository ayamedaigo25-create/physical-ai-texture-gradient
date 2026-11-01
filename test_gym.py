import numpy as np
from gym_texture_env import TextureGymEnv

env = TextureGymEnv(max_steps=10)

try:
    obs, info = env.reset(seed=42)

    print("=== ENVIRONMENT READY ===")
    print("Image:", obs["image"].shape)
    print("State:", obs["state"])
    print("Action space:", env.action_space)

    rng = np.random.default_rng(42)
    total_reward = 0.0

    for step in range(10):
        # 最初は小さなランダム運動、グリッパーは開放
        action = rng.uniform(
            -0.2, 0.2, size=5
        ).astype(np.float32)
        action[4] = -1.0

        obs, reward, terminated, truncated, info = (
            env.step(action)
        )

        total_reward += reward

        print(
            f"Step {step+1:02d} | "
            f"reward={reward:.4f} | "
            f"time={info['elapsed']:.2f}s | "
            f"cube={np.round(info['cube_xyz'], 3)}"
        )

        if terminated or truncated:
            break

    print("=== EPISODE COMPLETE ===")
    print("Total reward:", total_reward)
    print("Success:", info["success"])

finally:
    env.close()
