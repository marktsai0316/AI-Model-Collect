
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

# 建立帶有 Viewer 的環境
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

env = Monitor(env)

# 建立 PPO
model = PPO(
    policy="MlpPolicy",
    env=env,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    verbose=1,
    device="cpu",
    tensorboard_log="./logs/"
)

# 開始訓練
try:
    model.learn(
        total_timesteps=1_000_000,
        progress_bar=True
    )
finally:
    model.save("ppo_lunar")
    env.close()