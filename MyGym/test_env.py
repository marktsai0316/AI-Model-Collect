import gymnasium as gym

env = gym.make("LunarLander-v3", render_mode="human")

episode = 0

while True:
    observation, info = env.reset()
    total_reward = 0
    episode += 1
    print("Step : {episode}")
    while True:
        action = env.action_space.sample()

        observation, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if terminated or truncated:
            break

    # 檢查是否安全降落
    base_env = env.unwrapped

    success = (
        terminated
        and not base_env.game_over
        and not base_env.lander.awake
    )

    print(f"Episode: {episode}, Reward: {total_reward:.2f}")

    if success:
        print("成功安全降落！停止程式")
        break

    print("降落失敗，重新開始")

env.close()