#viewer.launch()（會阻塞主執行緒、適合直接看模擬）與 viewer.launch_passive()（不會阻塞主執行緒、適合在迴圈中自訂控制邏輯）兩種方式。


import mujoco
import mujoco.viewer



# 1. 定義自包含的 MJCF 模型（包含一個平面、一個方塊、一個球體與一個攝影機）
xml_string = """
<mujoco model="lidar_simulation_scene">
  <worldbody>
    <light pos="0 0 3" dir="0 0 -1"/>
    <geom type="plane" size="5 5 0.1" rgba="0.8 0.8 0.8 1"/>
    
    <!-- 障礙物 1：方塊 -->
    <body name="box" pos="1.2 0.0 0.3">
      <geom type="box" size="0.3 0.3 0.3" rgba="0.2 0.5 0.8 1"/>
    </body>
    
    <!-- 障礙物 2：球體 -->
    <body name="sphere" pos="-0.8 1.0 0.4">
      <geom type="sphere" size="0.4" rgba="0.8 0.3 0.3 1"/>
    </body>
    
    <!-- 虛擬 LiDAR / 深度相機 -->
    <body name="sensor_rig" pos="0 -2.0 0.8" euler="25 0 0">
      <camera name="lidar_cam" fovy="60"/>
    </body>
  </worldbody>
</mujoco>
"""

# 1. 載入你的 XML 模型檔案
# 請將 "your_model.xml" 替換成你的 xml 檔案路徑（例如內建模型或自訂路徑）
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# 2. 啟動互動式 Viewer 視窗
# 這行程式碼會開啟視窗並執行內建的物理迴圈
mujoco.viewer.launch(model, data)