import numpy as np
from gym_texture_env import TextureGymEnv

env = TextureGymEnv(max_steps=1)

try:
    for episode in range(3):
        obs, _ = env.reset(seed=42 if episode == 0 else None)

        action = np.array([0, 0, 0, 0, -1], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"Episode {episode + 1}: "
            f"reward={reward:.4f}, "
            f"terminated={terminated}, "
            f"truncated={truncated}",
            flush=True
        )

    print("=== 3 EPISODES COMPLETE ===")
finally:
    env.close()
