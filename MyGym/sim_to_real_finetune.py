import gymnasium as gym
from stable_baselines3 import PPO

# ==========================================
# 階段 1：在理想的「模擬器 (Sim)」中訓練基礎模型
# ==========================================
print("=== 階段 1：在標準模擬器中進行基礎訓練 (Pre-training) ===")
env_sim = gym.make("LunarLander-v3")

# 初始化 PPO 模型
model = PPO("MlpPolicy", env_sim, verbose=1, learning_rate=3e-4)

# 為了節省時間，這裡僅進行少量訓練（實際應用中會訓練數十萬步）
model.learn(total_timesteps=50_000)
model.save("lunar_base_model")
print("基礎模型已儲存！\n")

# ==========================================
# 階段 2：模擬「真機環境 (Reality)」(帶有物理偏差與干擾)
# ==========================================
# 假設真實世界的重力不同，且伴隨亂流（強風干擾）
print("=== 階段 2：切換至帶有物理偏差的「真機環境」===")
class RealWorldLunarLanderWrapper(gym.Env):
    """模擬真機環境：修改重力、增加風阻與亂流干擾"""
    def __init__(self):
        self.env = gym.make("LunarLander-v3", gravity=-8.0) # 真實重力與模擬不同
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # 模擬真實世界的突發側風干擾（對水平速度 x_dot 注入隨機雜訊）
        import numpy as np
        if np.random.rand() < 0.3:  # 30% 機率起風
            obs[2] += np.random.normal(0, 0.2) # 干擾水平速度
            
        # 真機獎勵懲罰：若因為環境偏差導致碰撞，給予額外懲罰
        if terminated and obs[1] > 0.1: # 尚未著陸就結束（摔毀）
            reward = float(reward)- 50.0 
            
        return obs, reward, terminated, truncated, info

env_real = RealWorldLunarLanderWrapper()

# ==========================================
# 階段 3：在「真機環境」上進行微調 (Fine-Tuning)
# ==========================================
print("=== 階段 3：載入模型，開始進行真機微調 (Fine-Tuning) ===")
# 載入剛剛在模擬器練好的模型
model = PPO.load("lunar_base_model", env=env_real)

# ⚠️ 關鍵：微調時通常必須「調降學習率」，避免破壞已經學好的基礎動作
model.learning_rate = 5.0e-5  # 比原本降低約 6 倍

# 讓模型直接在真機環境中互動並更新權重（適應真實風阻與重力偏差）
model.learn(total_timesteps=20_000)

# 儲存微調後的最終模型
model.save("lunar_real_adapted_model")
print("✅ 真機微調完成！適應後的模型已儲存。")