#!/usr/bin/env python3
"""play_live.py — 交互式推理 viewer，WASD 多模式切换（对齐 push_resist 的
interactive_combined_viewer 手感）。

microtest 是单个速度跟踪策略，模式不是"走路专家"而是速度指令 setpoint；
模式间用指令平滑 ramp 过渡（对应 combined viewer 的 blend）。无参考动作，
故没有相位对齐/参考锚定。

用法 (macOS 需用 mjpython 启动 viewer):
    mjpython play_live.py                                 # 自动用最新 model_final.pt
    mjpython play_live.py --checkpoint logs/<run>/model_2700.pt

按键:
    W / S     前进 / 后退       A / D     左移 / 右移     X         静止
    ↑/↓       vx ± 步进        ←/→       wz ± 步进 (原地转向)
    [ / ]     vy ± 步进        +/-       切换步进 (0.01 / 0.02 / 0.05)
    B         推力扰动开关      R 重置 | P 暂停/继续 | H 帮助 | Q/Esc 退出
"""
from __future__ import annotations

import argparse
import dataclasses
import glob
import os
import time

import mujoco.viewer
import numpy as np
import torch

from duck_vec_env import DuckVecEnv, MicroCfg
from play import build_policy

# 模式 -> 速度指令 setpoint (vx, vy, wz)
MODES = {
    "forward":  (0.35,  0.0,   0.0),
    "backward": (-0.25, 0.0,   0.0),
    "left":     (0.0,   0.05,  0.0),
    "right":    (0.0,  -0.05,  0.0),
    "standing": (0.0,   0.0,   0.0),
}
CMD_STEP_OPTS = [0.01, 0.02, 0.05]
CMD_LIMIT = {"vx": (-0.30, 0.40), "vy": (-0.05, 0.05), "wz": (-0.80, 0.80)}
RAMP_ALPHA = 1.0 / 20          # 每控制步向目标靠拢比例 (~0.4s 平滑过渡)
PUSH_ON_TOTAL_STEPS = 30_000_000  # 推力开启时推到课程末档使训练内置推力生效


def find_latest_checkpoint() -> str:
    roots = sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "micro_velocity_*")))
    if not roots:
        raise FileNotFoundError("logs 下没有 micro_velocity_* 训练目录")
    run = roots[-1]
    final = os.path.join(run, "model_final.pt")
    if os.path.exists(final):
        return final
    pts = sorted(glob.glob(os.path.join(run, "model_*.pt")), key=lambda p: int(p.rsplit("_", 1)[1].split(".")[0]))
    return pts[-1] if pts else final


def add_cmd_indicator(viewer, base_pos, cmd):
    """在躯干上方画绿色箭头，指示当前速度指令方向与大小。"""
    mag = float(np.linalg.norm(cmd[:2]))
    if mag < 1e-4:
        return
    scale = 3.0  # m/s -> 箭头长度 (m)
    length = float(np.clip(mag * scale, 0.05, 0.4))
    dir_vec = np.array([cmd[0], cmd[1], 0.0], dtype=np.float64) / mag
    from_pt = np.array([base_pos[0], base_pos[1], base_pos[2] + 0.10], dtype=np.float64)
    to_pt = from_pt + dir_vec * length
    geom = viewer.user_scn.geoms[0]
    mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_ARROW, 0.020, from_pt, to_pt)
    geom.rgba = (0.2, 0.9, 0.3, 0.9)
    viewer.user_scn.ngeom = 1


def main() -> None:
    parser = argparse.ArgumentParser(description="microtest WASD 多模式交互 viewer")
    parser.add_argument("--checkpoint", default=None, help="模型检查点 (缺省用最新日志的 model_final.pt)")
    args = parser.parse_args()
    checkpoint = args.checkpoint or find_latest_checkpoint()

    # 与 play.py 一致的干净推理配置：关观测噪声/编码器偏置/动作延迟/重置扰动
    cfg = dataclasses.replace(
        MicroCfg(),
        noise_joint_pos=0.0, noise_joint_vel=0.0,
        noise_gyro=0.0, noise_gravity=0.0,
        encoder_bias_range=0.0, action_delay_max=0,
        reset_joint_noise=0.0, reset_base_tilt_deg=0.0,
    )
    env = DuckVecEnv(cfg, num_envs=1, seed=42, device="cpu")
    policy = build_policy(checkpoint)

    state = {
        "mode": "standing",
        "target": np.array(MODES["standing"], dtype=np.float64),
        "cmd": np.zeros(3),
        "step_idx": CMD_STEP_OPTS.index(0.02),
        "push_on": False,
        "reset": False,
        "paused": False,
        "close": False,
    }

    def step_size() -> float:
        return CMD_STEP_OPTS[state["step_idx"]]

    def clamp(cmd):
        for i, k in enumerate(["vx", "vy", "wz"]):
            cmd[i] = float(np.clip(cmd[i], *CMD_LIMIT[k]))

    def set_mode(mode):
        state["mode"] = mode
        state["target"] = np.array(MODES[mode], dtype=np.float64)
        status("已切换")

    def status(prefix="当前"):
        c = state["cmd"]
        print(f"{prefix}: 动作={state['mode']}  vx={c[0]:+.3f}  vy={c[1]:+.3f}  wz={c[2]:+.3f}  "
              f"步进={step_size():.2f}  {'[推力]' if state['push_on'] else '[无推力]'}"
              f"{'  [暂停]' if state['paused'] else ''}")

    def help_text():
        print("\n模式: W前 S后 A左 D右 X静止 | ↑↓ vx  ←→ wz  [] vy  +/- 步进 | B 推力开关")
        print("      R 重置 | P 暂停/继续 | H 帮助 | Q/Esc 退出")
        status("当前")

    def set_push(on: bool):
        state["push_on"] = on
        env.total_steps = PUSH_ON_TOTAL_STEPS if on else 0
        env.push_timer[0] = 2.0 if on else 1e9
        status("推力已" + ("开启" if on else "关闭"))

    def on_key(key: int):
        s = step_size()
        if key in (ord("W"), ord("w")): set_mode("forward")
        elif key in (ord("S"), ord("s")): set_mode("backward")
        elif key in (ord("A"), ord("a")): set_mode("left")
        elif key in (ord("D"), ord("d")): set_mode("right")
        elif key in (ord("X"), ord("x")): set_mode("standing")
        elif key == 265: state["mode"] = "手动"; state["target"][0] += s
        elif key == 264: state["mode"] = "手动"; state["target"][0] -= s
        elif key == 263: state["mode"] = "手动"; state["target"][2] += s
        elif key == 262: state["mode"] = "手动"; state["target"][2] -= s
        elif key == ord("["): state["mode"] = "手动"; state["target"][1] += s
        elif key == ord("]"): state["mode"] = "手动"; state["target"][1] -= s
        elif key == ord("0"): set_mode("standing")
        elif key in (ord("+"), ord("=")): state["step_idx"] = min(len(CMD_STEP_OPTS) - 1, state["step_idx"] + 1)
        elif key in (ord("-"), ord("_")): state["step_idx"] = max(0, state["step_idx"] - 1)
        elif key in (ord("B"), ord("b")): set_push(not state["push_on"])
        elif key in (ord("R"), ord("r")): state["reset"] = True
        elif key in (ord("P"), ord("p")): state["paused"] = not state["paused"]
        elif key in (ord("H"), ord("h")): help_text(); return
        elif key in (ord("Q"), ord("q"), 256): state["close"] = True; return
        clamp(state["target"])
        status()

    help_text()
    obs = env.get_observations()

    with mujoco.viewer.launch_passive(env.models[0], env.datas[0], key_callback=on_key) as viewer:
        while viewer.is_running() and not state["close"]:
            tick = time.perf_counter()
            if state["reset"]:
                env.reset()
                state["reset"] = False
                state["cmd"] = np.zeros(3)
                print("已重置机器人。")

            # 指令平滑 ramp 到目标 setpoint，再每帧注入（避免 reset 后 _sample_commands 覆盖）
            state["cmd"] += RAMP_ALPHA * (state["target"] - state["cmd"])
            env.commands[0] = state["cmd"]

            if not state["paused"]:
                with torch.inference_mode():
                    action = policy.act_inference(obs)
                obs, rew, done, _ = env.step(action)
                if done.item():
                    print(f"机器人倒下/回合结束，自动重置。 (动作={state['mode']})")

            base_pos = env.datas[0].xpos[env.models[0].body("base").id]
            viewer.user_scn.ngeom = 0
            add_cmd_indicator(viewer, base_pos, state["cmd"])
            viewer.sync()

            delay = cfg.ctrl_dt - (time.perf_counter() - tick)
            if delay > 0:
                time.sleep(delay)


if __name__ == "__main__":
    main()
