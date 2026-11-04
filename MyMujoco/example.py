import matplotlib
matplotlib.use('TkAgg')  # 或者 'Qt5Agg'
#import matplotlib.pyplot as plt

import mujoco
import numpy as np
import matplotlib.pyplot as plt

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

# 載入模型與資料
model = mujoco.MjModel.from_xml_string(xml_string)
data = mujoco.MjData(model)

# 進行一次物理模擬步進更新場景
mujoco.mj_step(model, data)

# 2. 初始化 Renderer（設定解析度，例如 160x120 模擬稀疏點雲或輕量掃描）
width, height = 160, 120
renderer = mujoco.Renderer(model, height=height, width=width)

# 3. 取得相機資訊以計算投影反演
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "lidar_cam")
fovy = model.cam_fovy[cam_id]

# 計算焦距 (Focal Length)
f = (height / 2.0) / np.tan(np.radians(fovy / 2.0))

# 4. 渲染深度圖
renderer.enable_depth_rendering()
renderer.update_scene(data, camera="lidar_cam")
depth_normalized = renderer.render() # 範圍在 [0.0, 1.0]

# 將 MuJoCo 歸一化深度轉換為真實距離 (Meters)
extent = model.stat.extent
znear = model.vis.map.znear * extent
zfar = model.vis.map.zfar * extent
# OpenGL 投影公式反算深度
depth_meters = znear / (1.0 - depth_normalized * (1.0 - znear / zfar))

# 過濾掉沒有打到物體的背景（極遠處）
depth_meters[depth_normalized >= 0.999] = np.nan

# 5. 將 2D 深度圖反投影轉換為 3D 點雲 (Point Cloud in Camera Frame)
u_indices, v_indices = np.meshgrid(np.arange(width), np.arange(height))

# 轉換為相機座標系下的 (X, Y, Z)
# 注意：MuJoCo 相機座標系中，Z 軸朝向相機後方，X朝右，Y朝上（或視投影習慣而定）
X = (u_indices - width / 2.0) * depth_meters / f
Y = -(v_indices - height / 2.0) * depth_meters / f  # 翻轉 V 軸
Z = -depth_meters

# 塑形為 (N, 3) 陣列
points_cam = np.stack((X, Y, Z), axis=-1).reshape(-1, 3)
# 移除無效點 (NaN)
points_cam = points_cam[~np.isnan(points_cam).any(axis=1)]

# 6. 將點雲從相機座標系轉換到世界座標系 (World Frame)
# 取得相機在世界座標系中的位置與旋轉矩陣
cam_pos = data.cam_xpos[cam_id]          # (3,)
cam_mat = data.cam_xmat[cam_id].reshape(3, 3) # (3, 3)

# 座標轉換：P_world = R * P_cam + T
points_world = (cam_mat @ points_cam.T).T + cam_pos

print("成功生成 3D 點雲，總點數: {points_world.shape[0]}")

# 7. 使用 Matplotlib 進行 3D 視覺化繪製
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(projection='3d')

# 以 Z 軸高度作為顏色映射
sc = ax.scatter(
    points_world[:, 0], 
    points_world[:, 1], 
    points_world[:, 2], 
    c=points_world[:, 2], 
    cmap='viridis', 
    s=2
)

ax.set_title("MuJoCo Simulated LiDAR 3D Point Cloud")
ax.set_xlabel("X (Meters)")
ax.set_ylabel("Y (Meters)")
ax.set_zlabel("Z (Meters)")
ax.set_xlim([-3, 3])
ax.set_ylim([-3, 3])
ax.set_zlim([0, 3])
plt.colorbar(sc, label="Height (Z)")
plt.show()
#plt.savefig("lidar_pointcloud.png", bbox_inches="tight")
#plt.close()
#print("點雲圖已成功儲存為 lidar_pointcloud.png！")