import mujoco
import mujoco.viewer
import pygame

# 1. 初始化 Pygame 音效系統
pygame.mixer.init()

# 2. 載入並播放背景音樂 (BGM)
# -1 代表無限循環播放
try:
    pygame.mixer.music.load("background_music.mp3")
    pygame.mixer.music.set_volume(0.3)  # 設定音量 (0.0 到 1.0)
    pygame.mixer.music.play(-1)
except pygame.error:
    print("找不到背景音樂檔案，將以無音樂模式運行。")

# 3. 載入互動音效 (SFX)，例如碰撞聲或腳步聲
try:
    footstep_sound = pygame.mixer.Sound("footstep.wav")
except pygame.error:
    footstep_sound = None

# 載入 MuJoCo 模型
model = mujoco.MjModel.from_xml_path("your_model.xml")
data = mujoco.MjData(model)

with mujoco.viewer.launch_viewer(model, data) as viewer:
    step_counter = 0
    while viewer.is_running():
        mujoco.mj_step(model, data)
        
        # 範例：根據模擬狀態觸發音效
        # 例如：每隔固定步數模擬機器人觸地，或是透過檢查 data.contact 來判斷碰撞
        if footstep_sound and step_counter % 50 == 0:
            footstep_sound.play()
            
        step_counter += 1
        viewer.sync()

# 結束時清理音效資源
pygame.mixer.quit()