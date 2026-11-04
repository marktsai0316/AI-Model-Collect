#屬於依照預先設定節拍同步，不是自動分析音樂節奏。初次播放時也可能有少量音訊延遲。

import time
import math
import numpy as np
import mujoco
import mujoco.viewer
import pygame

XML = """
<mujoco model="Dance Robot">
  <option timestep="0.002" gravity="0 0 -9.81"/>

  <worldbody>
    <light pos="0 0 3"/>
    <geom type="plane" size="3 3 0.1"/>

    <body pos="0 0 0.6">
      <geom type="box" size=".15 .12 .2" mass="2"/>

      <body pos="0.15 0 0.1">
        <joint name="left" type="hinge"
               axis="0 1 0" range="-1 1"/>
        <geom type="capsule"
              fromto="0 0 0 0.35 0 0"
              size=".04" mass=".2"/>
      </body>

      <body pos="-0.15 0 0.1">
        <joint name="right" type="hinge"
               axis="0 1 0" range="-1 1"/>
        <geom type="capsule"
              fromto="0 0 0 -0.35 0 0"
              size=".04" mass=".2"/>
      </body>
    </body>
  </worldbody>

  <actuator>
    <position joint="left" kp="20"/>
    <position joint="right" kp="20"/>
  </actuator>
</mujoco>
"""

BPM = 120

model = mujoco.MjModel.from_xml_string(XML)
data = mujoco.MjData(model)

pygame.mixer.init()
pygame.mixer.music.load("music.mp3")

with mujoco.viewer.launch_passive(model, data) as viewer:

    pygame.mixer.music.play()
    start = time.perf_counter()

    while viewer.is_running():

        elapsed = time.perf_counter() - start
        beat = elapsed * BPM / 60.0

        # 左右關節交替擺動
        left = 0.6 * math.sin(2 * math.pi * beat)
        right = 0.6 * math.sin(
            2 * math.pi * beat + math.pi
        )

        data.ctrl[0] = left
        data.ctrl[1] = right

        mujoco.mj_step(model, data)
        viewer.sync()

        # 與真實時間同步
        next_step = start + data.time
        delay = next_step - time.perf_counter()

        if delay > 0:
            time.sleep(delay)

pygame.mixer.music.stop()
pygame.mixer.quit()