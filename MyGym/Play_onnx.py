import gymnasium as gym
import numpy as np
import onnxruntime as ort

# 1. 載入 ONNX 模型
session = ort.InferenceSession("ppo_lunar.onnx")
input_name = session.get_inputs()[0].name

# 2. 建立 LunarLander 環境
env = gym.make("LunarLander-v3", render_mode="human")
obs, _ = env.reset()

done = False
while not done:
  # 調整維度以符合 ONNX 輸入要求 (通常為 [1, observation_dim])
  obs_tensor = np.array([obs], dtype=np.float32)

  # 執行推論
  outputs = session.run(None, {input_name: obs_tensor})

  # 取得動作 (根據離散或連續動作空間處理)
  action = np.argmax(outputs[0])  # 若為離散動作

  obs, reward, terminated, truncated, info = env.step(action)
  done = terminated or truncated

env.close()
