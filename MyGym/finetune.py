import gymnasium as gym
from stable_baselines3 import PPO

# 1. 建立相同的 Gymnasium 環境
env = gym.make("LunarLander-v3", render_mode="human") # 或你的自定義環境

# 2. 載入前次儲存的模型，並將環境綁定上去
# 提示：如果你的環境做了修改（例如改變了 wrapper 或獎勵函數），確保與訓練時相容
model = PPO.load(
    "ppo_lunar", 
    env=env,
    verbose=1, 
    device="cpu"
)

# 開始訓練
try:
    # 3. 繼續訓練
    # reset_num_timesteps=False 確保 TensorBoard 的步數（timesteps）與學習率排程（learning rate schedule）不會被重置
    model.learn(
       total_timesteps=50000, 
       reset_num_timesteps=False, 
       progress_bar=True
    )

finally:
    # 4. 儲存微調後的模型
    model.save("ppo_lunar_finetuned")
    env.close()