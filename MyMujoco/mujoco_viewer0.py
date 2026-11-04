#python mujoco_viewer.py ./model.xml

import sys
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


    # 啟動互動式視覺化介面
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()