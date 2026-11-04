import gymnasium as gym
from stable_baselines3 import PPO

# 載入訓練好的 PPO 模型
model = PPO.load("ppo_lunar", device="cpu")

# 開啟視覺化模擬
env = gym.make(
    "LunarLander-v3",
    render_mode="human"
)

episode = 0

try:
    while True:
        observation, info = env.reset()
        total_reward = 0.0
        episode += 1

        while True:
            # PPO Policy 決定引擎動作
            action, _states = model.predict(
                observation,
                deterministic=True
            )

            # 執行模擬
            observation, reward, terminated, truncated, info = env.step(
                int(action)
            )

            #mark total_reward += reward
            total_reward += float(reward)

            print(
                f"Episode={episode} "
                f"Action={action} "
                f"Reward={reward:.2f}"
            )

            if terminated or truncated:
                break

        # 判斷是否真正安全降落
        base_env = env.unwrapped

        success = (
            terminated
            #and not base_env.game_over
            #and not base_env.lander.awake
            and not getattr(base_env, 'game_over', False)
            #and not getattr(getattr(base_env, 'game_over', False)
            and not getattr(getattr(base_env, "lander", None), "awake", False)
        )

        print(f"Total Reward: {total_reward:.2f}")

        if success:
            print("成功安全降落！")
            break

        print("降落失敗，重新開始")

finally:
    env.close()
