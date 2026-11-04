import sys
#python mujoco_viewer.py ./model.xml

import time
import mujoco
import mujoco.viewer

def main():
    # 檢查是否有傳入 xml 檔案路徑作為參數
    if len(sys.argv) < 2:
        print("錯誤：請提供 XML 模型檔案路徑！")
        print("使用方式: python mujoco_viewer.py <your_model.xml>")
        sys.exit(1)

    # 取得命令列的第一個參數作為 xml 路徑
    xml_path = sys.argv[1]
    print(f"正在載入模型: {xml_path}...")
    
    try:
        model = mujoco.MjModel.from_xml_path(xml_path)
    except Exception as e:
        print(f"載入 XML 失敗: {e}")
        sys.exit(1)

    data = mujoco.MjData(model)

    print("啟動 MuJoCo Viewer 視窗...")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # 執行一步物理模擬
            mujoco.mj_step(model, data)

            # 同步畫面到 viewer
            viewer.sync()

            # 維持與模型設定相同的物理時間步進頻率
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

if __name__ == "__main__":
    main()