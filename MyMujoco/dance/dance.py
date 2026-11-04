#uv add mujoco librosa pygame numpy

import time
import librosa
import mujoco
import mujoco.viewer
import numpy as np
import pygame

# 1. 載入音樂並分析節拍
audio_path = "song.mp3"
print(f"正在分析音樂: {audio_path} ...")
y, sr = librosa.load(audio_path)

# 萃取節拍時間點 (秒)
tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
beat_times = librosa.frames_to_time(beat_frames, sr=sr)
tempo_val = float(np.squeeze(tempo))
print(f"分析完成！BPM: {tempo_val:.1f}, 共找到 {len(beat_times)} 個節拍點。")

# 2. 初始化 Pygame 音效系統
pygame.mixer.init()
pygame.mixer.music.load(audio_path)
pygame.mixer.music.set_volume(0.5)

# 3. 載入 MuJoCo 模型 (確保 your_model.xml 與此指令檔在同一資料夾)
model = mujoco.MjModel.from_xml_path("your_model.xml")
data = mujoco.MjData(model)

next_beat_idx = 0
beat_impulse = 0.0  # 重拍爆發能量

print("啟動 MuJoCo 模擬與音樂同步舞蹈...")
pygame.mixer.music.play()
start_time = time.time()

with mujoco.viewer.launch(model, data) as viewer:
    while viewer.is_running():
        # 計算目前音樂播放的時間 (秒)
        current_audio_time = time.time() - start_time
        
        # 檢查是否到達下一個節拍點
        if next_beat_idx < len(beat_times) and current_audio_time >= beat_times[next_beat_idx]:
            beat_impulse = 1.0  # 當抵達重拍時，賦予瞬間爆發力
            next_beat_idx += 1

        # 讓重拍能量隨時間快速衰減（形成彈性律動感）
        beat_impulse = max(0.0, beat_impulse - 0.08)

        # -------------------------------------------------------------
        # 互動邏輯：將音樂節奏映射到 your_model.xml 的動作
        # -------------------------------------------------------------
        
        # A. 軀幹上下彈跳 (對應 XML 中的 torso freejoint，qpos[2] 為 Z 軸高度)
        base_z = 1.0  # 預設站立高度
        bounce = 0.06 * np.sin(current_audio_time * 8.0) + 0.15 * beat_impulse
        data.qpos[2] = base_z + bounce

        # B. 雙腿關節交替擺動 (透過 actuator 控制 right_hip 與 left_hip)
        # 利用正弦波製造律動，並在重拍時加入額外擺幅
        leg_swing = 0.4 * np.sin(current_audio_time * 6.0)
        
        if model.nu >= 2:
            # 右腿與左腿反向擺動，並在重拍時用力踢一下
            data.ctrl[0] = leg_swing + (0.25 * beat_impulse)   # right_hip_motor
            data.ctrl[1] = -leg_swing - (0.25 * beat_impulse)  # left_hip_motor

        # 執行物理模擬步進並同步畫面
        mujoco.mj_step(model, data)
        viewer.sync()

pygame.mixer.quit()