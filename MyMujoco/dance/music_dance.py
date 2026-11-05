#!/usr/bin/env python3

#sudo apt update
#sudo apt install -y libportaudio2 portaudio19-dev


#uv add librosa sounddevice numpy mujoco
#python 3.13


"""Beat-synchronised MicroDuck dance in CPU MuJoCo (simulation only)."""




import argparse
import time
from pathlib import Path

import librosa
import mujoco
import mujoco.viewer
import numpy as np
import sounddevice as sd

JOINTS = (
    'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle',
    'neck_pitch', 'head_pitch', 'head_yaw', 'head_roll',
    'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle',
)

def find_scene(root):
    base = root / 'test/microduck_rl-develop/src/mjlab_microduck/robot/microduck'
    #base = './microduck_rl-develop/src/mjlab_microduck/robot/microduck'
    for name in ('scene.xml', 'scene_walk.xml', 'scene_groundcontact.xml'):
        p = base / name
        if p.exists():
            return p
    raise FileNotFoundError(f'No scene XML in {base}; specify --scene')

def analyze(path):
    y, sr = librosa.load(str(path), sr=22050, mono=True)
    hop = 256
    onset = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=sr, hop_length=hop, trim=False)
    bpm = float(np.asarray(tempo).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop)
    rms = librosa.feature.rms(y=y, hop_length=hop)[0]
    rms_times = librosa.frames_to_time(np.arange(len(rms)), sr=sr, hop_length=hop)
    rms /= max(float(np.percentile(rms, 95)), 1e-8)
    return bpm, beat_times, rms_times, np.clip(rms, 0, 1), len(y) / sr

def pose(model, data):
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, 'STAND')
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        mujoco.mj_resetData(model, data)
        print('WARNING: no STAND keyframe; using qpos0')
    mujoco.mj_forward(model, data)
    baseline = {}
    for j in JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid < 0:
            raise ValueError(f'Missing joint: {j}')
        baseline[j] = float(data.qpos[model.jnt_qposadr[jid]])
    actuators = {}
    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        if jid < 0:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, jid)
        if name in baseline:
            actuators[name] = i
    missing = set(JOINTS) - set(actuators)
    if missing:
        raise ValueError(f'Missing position actuators for {sorted(missing)}')
    return baseline, actuators

def command(model, data, base, actuators, offsets):
    for name, i in actuators.items():
        target = base[name] + offsets.get(name, 0.0)
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            target = np.clip(target, lo + 0.01, hi - 0.01)
        if model.actuator_ctrllimited[i]:
            lo, hi = model.actuator_ctrlrange[i]
            target = np.clip(target, lo, hi)
        data.ctrl[i] = target

def dance_offsets(t, beats, rms_t, rms, scale):
    # Beat phase tracks actual detected beats rather than assuming fixed tempo.
    if len(beats) >= 2:
        k = int(np.searchsorted(beats, t, side='right') - 1)
        k = max(0, min(k, len(beats) - 2))
        period = max(0.2, float(beats[k + 1] - beats[k]))
        phase = 2 * np.pi * (t - beats[k]) / period
        beat_number = k
    else:
        phase = 2 * np.pi * t * 2
        beat_number = int(t * 2)
    energy = float(np.interp(t, rms_t, rms))
    strength = scale * (0.4 + 0.6 * energy)
    side = np.sin(phase / 2 + beat_number * np.pi / 2)
    bounce = (1 - np.cos(phase)) / 2
    twist = np.sin(phase)
    o = {
        'left_hip_roll': 0.07 * side,
        'right_hip_roll': 0.07 * side,
        'left_hip_yaw': 0.10 * twist,
        'right_hip_yaw': -0.10 * twist,
        'left_hip_pitch': 0.06 * bounce,
        'right_hip_pitch': -0.06 * bounce,
        'left_knee': -0.045 * bounce,
        'right_knee': 0.045 * bounce,
        'left_ankle': -0.045 * bounce,
        'right_ankle': 0.045 * bounce,
        'neck_pitch': 0.07 * bounce,
        'head_pitch': -0.09 * bounce,
        'head_yaw': 0.18 * side,
        'head_roll': 0.10 * side,
    }
    return {k: v * strength for k, v in o.items()}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('music', type=Path)
    ap.add_argument('--scene', type=Path)
  
    ap.add_argument('--scale', type=float, default=8.0)
    ap.add_argument('--offset', type=float, default=0.0, help='Audio timing correction in seconds')
    args = ap.parse_args()
    if not args.music.is_file():
        ap.error(f'Music not found: {args.music}')
    root = Path(__file__).resolve().parents[1]
    scene = args.scene or find_scene(root)
    print('Analyzing:', args.music)
    bpm, beats, rms_t, rms, duration = analyze(args.music)
    print(f'BPM={bpm:.1f}, beats={len(beats)}, duration={duration:.1f}s')
    model = mujoco.MjModel.from_xml_path(str(scene))

    # 關閉重力
    #model.opt.gravity[:] = [0.0, 0.0, 0.0]
    model.opt.gravity[:] = 0

    # 找到 freejoint 所在的 Root Body
    root_body_id = -1

    for j in range(model.njnt):
       if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE:
          root_body_id = int(model.jnt_bodyid[j])
          break

    if root_body_id < 0:
       raise RuntimeError("找不到 Root freejoint")

    # 將 Root Body 固定到 World
    spec = mujoco.MjSpec.from_file(str(scene))

    # 在模型編譯前加入 weld
    root_name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, root_body_id
    )

    weld = spec.add_equality()
    weld.type = mujoco.mjtEq.mjEQ_WELD
    weld.name = "root_fixed"
    weld.objtype = mujoco.mjtObj.mjOBJ_BODY
    weld.name1 = root_name
    weld.name2 = "world"

    model = spec.compile()


    data = mujoco.MjData(model)
    base, actuators = pose(model, data)
    # The project scene has position actuators; do not use this on motor/voltage XML.
    for name, i in actuators.items():
        if model.actuator_dyntype[i] != mujoco.mjtDyn.mjDYN_NONE:
            raise RuntimeError(f'Unsupported actuator dynamics for {name}')
    audio, sr = librosa.load(str(args.music), sr=44100, mono=False)
    audio = np.asarray(audio.T, dtype=np.float32) if audio.ndim == 2 else np.asarray(audio, dtype=np.float32)
    audio = np.ascontiguousarray(audio)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        print('Viewer ready. Playing audio...')
        # Playback begins close to clock zero; --offset can compensate device latency.
        sd.play(audio, sr, blocking=False)
        start = time.perf_counter()
        next_tick = start
        dt = model.opt.timestep
        try:
            while viewer.is_running():
                now = time.perf_counter()
                elapsed = now - start
                if elapsed >= duration:
                    break
                # Advance sim to wall-clock time; cap catch-up work after a stall.
                target = max(0.0, elapsed + args.offset)
                steps = 0
                while data.time < target and steps < 50:
                    offsets = dance_offsets(data.time, beats, rms_t, rms, args.scale)
                    command(model, data, base, actuators, offsets)
                    mujoco.mj_step(model, data)
                    steps += 1
                if steps == 50 and data.time < target:
                    # Skip stale simulation backlog after a GUI stall.
                    print('Simulation is lagging; reduce load or use a larger timestep')
                viewer.sync()
                next_tick += 1 / 60
                delay = next_tick - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                else:
                    next_tick = time.perf_counter()
        finally:
            sd.stop()
    print('Finished')

if __name__ == '__main__':
    main()
