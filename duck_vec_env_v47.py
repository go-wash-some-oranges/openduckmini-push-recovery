#!/usr/bin/env python3
"""DuckVecEnv — Open Duck Mini v2 向量化 CPU MuJoCo 训练环境。

设计遵循 microduck_rl (pollen-robotics) 的训练手册，详见 TRAINING_PLAN.md：

1. 纯任务奖励（速度指令跟踪），无 AMP / 参考动作；
2. 固定观测契约：51D actor / 55D critic，扩展只能加槽不能删槽；
3. 统一符号约定：惩罚函数返回 >=0 代价 × 负权重（日志中惩罚项必须 <=0）；
4. 无 jackpot 奖励：所有跟踪奖励为有界高斯；
5. 零指令显式采样（站立行为必须被训练）；
6. 域随机化在重置时"恢复-再-施加"，绝不跨回合累积；
7. 课程按 PPO 更新次数推进：指令范围渐宽、action-rate 税后置、推力晚引入；
8. 关节编码器偏置加在观测上 —— 策略看到的就是真机编码器视图。

N 个独立 MuJoCo CPU 实例，无需 GPU / JAX。
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

XML_PATH = Path(__file__).resolve().parent / "assets" / "xmls" / "scene_flat_terrain.xml"

JOINT_NAMES = [
    "left_hip_yaw", "left_hip_roll", "left_hip_pitch", "left_knee", "left_ankle",
    "neck_pitch", "head_pitch", "head_yaw", "head_roll",
    "right_hip_yaw", "right_hip_roll", "right_hip_pitch", "right_knee", "right_ankle",
]

NUM_STEPS_PER_ENV = 24  # 与 train.py / microduck_rl 一致，用于换算课程节奏
# 站立任务只关注腿/髋/膝/踝（microduck _LEG_JOINTS 布局一致）
LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]


@dataclass(frozen=True)
class MicroCfg:
    task_mode: str = "velocity"  # stand, push_recovery, velocity
    # ---- 时序 ----
    sim_dt: float = 0.002
    ctrl_dt: float = 0.02
    decimation: int = 10
    episode_length: int = 1000          # 20 s
    # ---- 动作 ----
    action_scale: float = 0.5
    max_motor_velocity: float = 5.24    # rad/s (STS3215 带载)
    # ---- 观测噪声 (1σ) ----
    noise_joint_pos: float = 0.01
    noise_joint_vel: float = 0.15
    noise_gyro: float = 0.05
    noise_gravity: float = 0.03
    encoder_bias_range: float = 0.015   # ±0.86°
    action_delay_max: int = 1           # 0..1 控制步
    # ---- 域随机化 ----
    mass_scale_range: tuple = (0.95, 1.05)
    friction_damping_range: tuple = (0.8, 1.2)
    armature_range: tuple = (0.9, 1.1)
    reset_joint_noise: float = 0.03
    reset_base_tilt_deg: float = 0.0   # v8 实战值
    push_interval_s: tuple = (1.2, 2.5)
    # ---- 终止 ----
    # gx^2+gy^2 超过即判定摔倒。0.30 ≈ 33° 太严：机器人稍倾斜就被重置，
    # 根本没机会走两步恢复。放宽到 0.55 ≈ 48°，给足“被推后迈步调整”的空间。
    fall_tilt_cost: float = 0.55
    min_height: float = 0.07
    max_height: float = 0.35
    # ---- 奖励权重 ----
    w_track_lin: float = 2.0         # v3: 对齐 microduck (2.0, std²=0.1/0.5)
    w_track_ang: float = 2.0
    sigma_track_lin: float = 0.316
    sigma_track_ang: float = 0.707
    w_alive: float = 0.5                # 每秒
    w_fall: float = -1.0                # 一次性
    w_base_height: float = -6.0
    base_height_target: float = 0.16
    base_height_low: float = 0.13
    base_height_high: float = 0.18
    w_base_height_low: float = -30.0
    w_base_height_high: float = -5.0
    w_orientation: float = -1.0
    w_lin_vel_z: float = -2.0
    w_ang_vel_xy: float = -0.1
    w_joint_limits: float = -1.0
    soft_joint_pos_limit_factor: float = 0.95
    # ---- stand 任务（microduck standup 配方）----
    STAND_Z: float = 0.15                # HOME 姿态实测基座高
    w_height_gauss: float = 1.0          # 宽高斯（远距牵引）
    w_height_gauss_sharp: float = 1.0    # 窄高斯（末段强梯度）
    height_std: float = 0.04
    height_std_sharp: float = 0.015
    w_height_l1: float = 7.5             # 高度 L1，逼起身，破除蹲姿盆地
    w_com_upward: float = 0.75           # 奖励 vz>0
    com_upward_max_height: float = 0.155
    w_upright_linear: float = 1.5        # cos(tilt) 粗牵引
    w_upright_sharp: float = 1.5         # 高度门控锐化直立
    upright_sharp_std: float = 0.3
    w_standing_composite: float = 3.75   # 高度×直立×姿态组合分
    w_pose_stand: float = 2.0            # 姿态匹配 HOME 腿关节
    pose_std: float = 0.5
    w_pose_l1: float = 1.25              # 姿态 L1 惩罚
    stand_pose_std: float = 0.4          # 组合分中的姿态 std
    w_head_pose_stand: float = 6.0       # 站立时头部保持参考抬头姿态
    w_head_pose_l1: float = 1.5          # 头部偏离参考姿态的线性约束
    head_pose_std: float = 0.25
    # 参考动作(150 条)标定的头部姿态：neck_pitch=+0.26, head_pitch=-0.26, yaw=0, roll=0
    stand_head_target: tuple = (0.26, -0.26, 0.0, 0.0)
    stand_head_joint_weights: tuple = (1.0, 0.8, 1.5, 1.5)
    w_head_limit: float = -2.0
    # ---- v2 步态整形（取自 microduck_rl 配方）----
    w_air_time: float = 3.0          # 抬脚离地时长奖励（指令门控）
    air_time_min: float = 0.10       # 步行摆动窗口 [0.10, 0.25] s
    air_time_max: float = 0.25
    w_foot_swing: float = -1.0       # 摆动期抬脚不足惩罚
    foot_swing_target: float = 0.02  # 抬脚目标 2 cm
    w_upright: float = 1.0           # 直立奖励 exp(-tilt²/0.05)
    upright_std2: float = 0.05
    w_stillness: float = 0.5         # 零指令时静止奖励
    stillness_std2: float = 0.01
    w_foot_clearance: float = -0.5   # 低脚位滑动惩罚（逼抬脚）
    foot_clearance_z: float = 0.03
    # ---- push_recovery 恢复激励（取自 microduck_rl 配方）----
    w_upright_progress: float = 1.25     # 势形奖励 Δcos(tilt)（v8 实战值）
    recovery_tilt_deg: float = 8.0       # 超过该倾斜进入“恢复态”，激励迈步换重心
    w_recovery_step: float = 3.0         # 恢复态下单脚抬脚奖励（v8 实战值）
    recovery_upright_deg: float = 5.0    # 直立进度奖励仅在超出该倾斜后计（避免原地乱动刷分）
    w_recover_walk: float = 0.0          # 关闭：v8→v16 对照证明该奖励激发蹭步
    recover_walk_vmax: float = 0.6       # 归一化目标速度（m/s）
    # ---- v17：弱方向定向 + 防漂移（推力幅度/频率保持 v8 不变）----
    weak_dir_prob: float = 0.35          # 推力方向按近期失败率加权的比例
    weak_dir_topk: int = 3               # 参与加权的失败率最高方向桶数
    push_mag_min_stage: int = 0          # 从第几课程段起启用方向加权
    # v25 证明了：0°/180° 等概率专项采样能训出后向卸力，但会牺牲前向刚度
    # （v25_16775：后向 0.55 从 42%→100%，前向 0.70 从 98%→82%）。
    # v26 从 v25_16775 出发，把专项采样重新压回前向：保住已有的向前卸力迈步，
    # 同时留少量后向样本防止遗忘。
    # v32: 前向仍占大头，后向预算 0.10→0.15 防遗忘（v28 曾用 0.30 有效但伤前向）。
    # v35: 从 v34_17525（后向 0.55=78/0.60=50，站姿 2.26°）出发，
    # 前向采样占比拉回 0.65 —— 目标是把前向 0.70 从 70 抬回 90+，
    # 同时保留已训出的后向门控让位（后向 0.15 防遗忘）。
    # v37: 从 v35_17700（前向 98/82、后向 0.55=88，但 0.60 掉到 22）出发。
    # 诊断（step_profile）：0.60 档位移 0.117m、峰值倾角 33° —— 现有"让位走"
    # 在更高冲量下退化成被推着滑走，支撑面永远追不上重心。新增 catch_step
    # （垫步接住）奖励治它，并把让位走降权。
    # v44 后向专项（从零训练）：采样预算压倒性投向后方。前向只留 10% 防完全遗忘
    # —— 前向能力后续靠与 v35 结合来补，本轮唯一目标是训出"会向后卸力"的后向专家。
    # v47 = v45 的配比（fwd 0.55 / bwd 0.25）+ v46 的势函数塑形。
    # 这是"势函数塑形到底有没有用"的干净对照：v45（无塑形）前向回得来、后向丢光；
    # v47（有塑形）如果两样都能保住，说明发现 1 是对的。
    fwd_sector_prob: float = 0.55          # 0° 扇区采样占比（把前向拿回来）
    bwd_sector_prob: float = 0.25          # 180° 扇区采样占比（守住后向成果）
    sector_halfwidth_deg: float = 22.5
    # v32: 后象限推力定向放大（v28 机制，幅度温和）。起点 v26_17100 后向 0.60=45%，
    # 定向加压逼出制动而不过度牺牲前向。
    bwd_push_boost: float = 1.15           # 后象限内推力缩放（其余方向 1.0）
    bwd_boost_halfwidth_deg: float = 45.0  # 以 180° 为中心的作用半宽
    # 推力门控的卸力迈步：参考 microduck 的 air_time/single_support 门控思路，
    # 只在「刚被推」的短窗口内给迈步/让位计分，避免全程刷分训出蹭步。
    # v32: 卸力窗口 0.30→0.45s、权重全面上调 —— 明确奖励"被推时让位换重心"而非硬顶。
    # 参考: v26 靠这套机制把后向 0.60 从 0 拉到 45%；本次是同一机制的加强版。
    push_react_window_s: float = 0.45
    w_yield_step: float = 4.0              # 窗口内单脚抬脚（换脚卸力，方向无关的安全机制）
    # ---- v46：势函数塑形（microduck 发现 1）----
    # microduck 原文："ANY positive reward for BEING in a fallen-ish state gets
    # farmed from some comfortable pose... POTENTIAL-BASED (Δcos tilt): rising
    # pays, falling costs, holding anything pays zero."
    # 我们 v26–v45 的 yield_step/yield_walk/catch_step 都是"状态给分"，
    # 在"已经开始倒但还没到 42°上限"的区间照样发满分 —— v35/v44 的失败轨迹
    # 正是"抬脚后倾角 12°→34° 一路拿分再摔"。这里加三重门控：
    w_react_upright: float = 6.0           # 窗口内直立势函数额外权重（把身体弄直才拿大分）
    yield_recover_only: bool = True        # 只在"倾角没有变大"的步给卸力分
    recover_gate_eps: float = -1.0e-4      # up_progress 门限（<-eps 视为正在倒）
    w_tilt_over: float = -8.0              # 窗口内倾角超阈值的持续惩罚（每秒）
    tilt_over_deg: float = 25.0            # 超过该倾角开始扣分（"停在倾斜态"不再免费）
    # v37: 让位走 2.5 → 1.0。高幅度下它奖励的是"被推着滑走"（0.60 档位移 0.117m
    # 仍拿满分），正是后向 0.55→0.60 断崖的元凶；降到 1.0 只保留"不硬顶"的引导。
    # v44: 1.0 → 2.5。用户要的是"看得见地向后让位"，v38 的沿推力位移只有 5.7cm
    # （基本原地硬扛）。加大让位走 + 垫步接住，让重心真的交给后落的支撑脚。
    w_yield_walk: float = 2.5              # 窗口内顺着推力方向让位
    # v38 长悬空惩罚：0.60 档失败轨迹显示，策略"抬脚卸力"后脚在空中停太久，
    # 支撑力矩消失、倾角单调涨到 38° 才摔（对照 v26 是双脚不离地、倾角自动衰减）。
    # 允许的正常换脚约 0.10s，超过 0.12s 起罚。
    # v44: 免罚时长 0.12 → 0.16s、权重 -5 → -3。原设置把"换脚"窗口掐得太短，
    # 策略只能原地抖一下脚；放开后允许真正的"后退一步再落稳"。
    w_react_flight: float = -3.0           # 窗口内单脚悬空超时的惩罚（每秒，逐脚累加）
    react_flight_free_s: float = 0.16      # 免罚的悬空时长
    # v38 恢复力矩加倍：窗口内倾角"正在回落"的奖励加倍（原本 w_upright_progress
    # 只有 1.25，被 yield_step=4.0 淹没），让"把身体拉回来"成为窗口内最强信号。
    react_up_boost: float = 3.0            # 窗口内 upright_progress 的额外倍率
    # v37 垫步接住：推力窗口内落地的脚，沿推力方向相对"被推瞬间的基座"的偏移
    # 越大越给分 —— 直接编码"退一步垫住重心"，治被推着漂。
    w_catch_step: float = 5.0              # 窗口内落地脚踩到推力方向一侧的奖励
    catch_step_ref: float = 0.08           # 落点偏移参考（米），达到即满分
    yield_tilt_max_deg: float = 42.0       # 超过该倾角视为已失控，不再给卸力分
    # v32: 非推力窗口内的抬脚惩罚（v28 机制，权重更温和）。
    # v28 用它治好了 v27a 的原地踏步退化，-3.0 把 16s 抬脚从 116 砍到 44；
    # 这里只取 -2.0，避免把策略压成"死扛"（v29 强压到 -6 的后遗症）。
    w_idle_lift: float = -2.0              # 窗口外单脚离地的惩罚（每秒，逐脚累加）
    w_play_still: float = -1.5           # 直立时基座水平速度平方惩罚（防"被推就走"退化）
    play_still_tilt_deg: float = 6.0     # 倾角低于该值视为"应原地站住"
    w_play_drift: float = -1.0           # 基座水平位移平方惩罚
    play_drift_cap: float = 0.15         # 位移惩罚上限（米），超出不再加罚，避免僵硬摔倒
    # ---- jump 任务：原地双脚跳跃 ----
    # 权重设计：跳的奖励必须压过"站着收分"的基线，否则策略只会原地站。
    # 核心信条（来自脚本）：蹲下蓄力、起身一定要快。
    # 结构：rise/飞行是引导小奖励；大分押在"一次完整跳跃的最高点"—— 只结算 >2cm 的真跳。
    jump_target_height: float = 0.008
    jump_min_flight_steps: int = 3
    jump_w_takeoff: float = 12.0      # 起身爆发(快)：只有真·脱离速度(≥~0.85m/s)才给大分 = "起身一定要快"
    jump_w_flight: float = 8.0        # 双脚离地后的腾空高度
    jump_w_apex: float = 8.0          # 腾空高度（飞行中持续给）
    jump_w_cycle: float = 30.0        # 一次完整真跳(>3cm离地)落地结算，按最高点比例
    jump_min_cycle_apex: float = 0.03 # 最低离地高度(米)，低于此不计为真跳(小抖腿清零)
    jump_w_landing: float = 2.0       # 落地软奖励（闭环）
    jump_w_impact: float = -0.15      # 硬着陆惩罚
    jump_w_crouch: float = 1.0        # 下蹲蓄力(弱引导：深蹲时向下才给少量,不足以刷分→逼向起身)
    jump_crouch_depth: float = 0.04   # 蓄力参考深度（相对 STAND_Z 下探多少算满蹲）
    jump_rise_thresh: float = 0.85    # 起身爆发阈值(m/s)：向上速度超过它才按爆发送分,小抖腿≈0
    jump_rise_k: float = 20.0         # 爆发 sigmoid 陡度
    jump_rise_kref: float = 0.5       # 连续梯度版爆发参考速度(m/s)：v/(v+ref) 单调增,全程有梯度
    jump_knee_compress_ref: float = 0.35  # 双膝相对 HOME 的屈曲参考深度(rad)
    jump_knee_ready_ratio: float = 0.45   # 达到参考屈曲深度的比例才算蓄力完成
    jump_knee_extend_ref: float = 4.0     # 膝伸展速度参考(rad/s)
    jump_w_knee_extend: float = 2.0       # 已蓄力后快速打开膝盖的辅助奖励
    jump_max_height: float = 0.45
    # ---- 指令采样（stage 0 初始值，课程会覆盖） ----
    zero_command_prob: float = 0.5
    vx_range: tuple = (-0.02, 0.05)
    vy_max: float = 0.02
    wz_max: float = 0.3


# 课程：(起始更新次数, action_rate权重, vx范围, vy上限, wz上限, 零指令比例, 推力上限)
CURRICULUM = [
    (0,     0.0,  (-0.02, 0.05), 0.02, 0.3, 0.50, 0.00),
    (150,  -0.1,  (-0.05, 0.10), 0.03, 0.4, 0.35, 0.00),
    (400,  -0.25, (-0.10, 0.20), 0.04, 0.6, 0.25, 0.20),
    (800,  -0.8,  (-0.20, 0.30), 0.05, 0.8, 0.15, 0.30),
    (1200, -1.2,  (-0.30, 0.40), 0.05, 0.8, 0.10, 0.40),
    (1500, -1.2,  (-0.30, 0.40), 0.05, 0.8, 0.10, 0.50),
    # v47 接力 ramp（自 v44_1800 resume，落在 1800 行）：前向直顶 0.70，
    # 后向经 _push_scale_for_angle 折减 ≈0.63。后向 0.60 以上"硬顶"物理必摔
    # （phys_lim 实测零动作峰值 40°+），所以这一档只能靠垫步卸力学会。
    (1900, -1.2,  (-0.30, 0.40), 0.05, 0.8, 0.10, 0.56),
    (2200, -1.2,  (-0.30, 0.40), 0.05, 0.8, 0.10, 0.64),
    (2600, -1.2,  (-0.30, 0.40), 0.05, 0.8, 0.10, 0.70),
]


def quat_to_rot(q: np.ndarray) -> np.ndarray:
    """(N,4) 四元数 (w,x,y,z) -> (N,3,3) 旋转矩阵（body->world）。"""
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    r = np.empty((q.shape[0], 3, 3), dtype=q.dtype)
    r[:, 0, 0] = 1 - 2 * (y * y + z * z)
    r[:, 0, 1] = 2 * (x * y - w * z)
    r[:, 0, 2] = 2 * (x * z + w * y)
    r[:, 1, 0] = 2 * (x * y + w * z)
    r[:, 1, 1] = 1 - 2 * (x * x + z * z)
    r[:, 1, 2] = 2 * (y * z - w * x)
    r[:, 2, 0] = 2 * (x * z - w * y)
    r[:, 2, 1] = 2 * (y * z + w * x)
    r[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return r


class DuckVecEnv:
    """rsl_rl 兼容的向量化环境接口（TensorDict 观测）。"""

    def __init__(self, cfg: MicroCfg, num_envs: int, seed: int = 42, device: str = "cpu"):
        self.cfg = cfg
        self.num_envs = num_envs
        self.device = device
        self.rng = np.random.default_rng(seed)

        self.models = [mujoco.MjModel.from_xml_path(str(XML_PATH)) for _ in range(num_envs)]
        self.datas = [mujoco.MjData(m) for m in self.models]
        m0 = self.models[0]

        # HOME 帧：直接取 XML 中的 home keyframe（与 microduck 的 HOME 概念一致）
        key_id = mujoco.mj_name2id(m0, mujoco.mjtObj.mjOBJ_KEY, "home")
        assert key_id >= 0, "scene XML 缺少 home keyframe"
        self.home_qpos = m0.key_qpos[key_id].copy()
        self.home_ctrl = m0.key_ctrl[key_id].copy()
        assert m0.nu == 14, f"期望 14 个执行器，实际 {m0.nu}"
        self.num_actions = m0.nu
        self.max_episode_length = cfg.episode_length

        # 关节软限位（qpos 侧）
        jlow, jup = m0.jnt_range[1:].T
        mid = 0.5 * (jlow + jup)
        half = 0.5 * (jup - jlow)
        f = cfg.soft_joint_pos_limit_factor
        self.soft_low = mid - f * half
        self.soft_up = mid + f * half

        # 传感器地址
        self.gyro_adr = int(m0.sensor("gyro").adr)
        self.linvel_adr = int(m0.sensor("local_linvel").adr)

        # v2: 脚底碰撞 geom -> 0/1，脚部 site
        self._foot_geom_idx = {
            int(m0.geom("left_foot_bottom_tpu").id): 0,
            int(m0.geom("right_foot_bottom_tpu").id): 1,
        }
        self._foot_site_ids = (int(m0.site("left_foot").id), int(m0.site("right_foot").id))

        # DR 基线（恢复-再-施加）
        self._body_mass0 = m0.body_mass.copy()
        self._frictionloss0 = m0.dof_frictionloss.copy()
        self._damping0 = m0.dof_damping.copy()
        self._armature0 = m0.dof_armature.copy()

        # 每环境缓冲
        self.last_action = np.zeros((num_envs, 14), dtype=np.float32)
        self.prev_ctrl = np.tile(self.home_ctrl, (num_envs, 1))
        self.desired_queue = np.tile(self.home_ctrl, (num_envs, 1))
        self.action_delay = np.zeros(num_envs, dtype=np.int32)
        self.encoder_bias = np.zeros((num_envs, 14), dtype=np.float32)
        self.commands = np.zeros((num_envs, 3), dtype=np.float32)
        self.push_timer = np.full(num_envs, 1e9, dtype=np.float32)
        self.push_active = np.zeros(num_envs, dtype=np.int32)    # 持续推力剩余控制步数
        self.push_ang = np.zeros(num_envs, dtype=np.float32)     # 活跃推力方向
        self.push_dv = np.zeros(num_envs, dtype=np.float32)      # 本次推力总冲量 (m/s)
        self.push_ang_log = np.zeros(num_envs, dtype=np.float32)  # 最近推力方向
        self.push_had = np.full(num_envs, False, dtype=bool)      # 本回合是否已被推过
        self.push_react = np.zeros(num_envs, dtype=np.float64)   # 推力反应窗口剩余时长(s)
        self.push_fail_hist = np.zeros(24, dtype=np.float64)      # 24 方向桶摔倒计数
        self.push_try_hist = np.zeros(24, dtype=np.float64)       # 24 方向桶尝试计数
        self.base_xy_start = np.zeros((num_envs, 2), dtype=np.float64)
        self.push_base_xy = np.zeros((num_envs, 2), dtype=np.float64)   # v37: 被推瞬间的基座 xy
        self.push_scale = 1.0
        self.push_step_dv = np.zeros(num_envs, dtype=np.float32) # 每控制步加速度 (m/s)
        self.foot_air_time = np.zeros((num_envs, 2), dtype=np.float64)
        self.foot_contact_prev = np.ones((num_envs, 2), dtype=bool)
        self.foot_z_lift = np.full((num_envs, 2), 0.02, dtype=np.float64)
        self.foot_xy_prev = np.zeros((num_envs, 2, 2), dtype=np.float64)
        self.episode_length_buf = np.zeros(num_envs, dtype=np.int64)
        self.reward_episode_sum = np.zeros(num_envs, dtype=np.float64)
        self._tilt_cos_prev = np.ones(num_envs, dtype=np.float64)
        self._jump_phase = np.zeros(num_envs, dtype=np.int8)  # 0 ground, 1 flight, 2 landing
        self._jump_peak = np.full(num_envs, self.cfg.STAND_Z, dtype=np.float64)
        self._jump_flight = np.zeros(num_envs, dtype=np.int64)
        self._jump_cycles = np.zeros(num_envs, dtype=np.int64)
        self._base_z_maxseen = np.full(num_envs, self.cfg.STAND_Z, dtype=np.float64)
        self._foot_both_air_prev = np.zeros(num_envs, dtype=bool)
        self._jump_charged = np.zeros(num_envs, dtype=bool)

        self.total_steps = 0
        self._finished_r: list[float] = []
        self._finished_l: list[int] = []

        # 观测缓存
        self._obs_policy = np.zeros((num_envs, 51), dtype=np.float32)
        self._obs_critic = np.zeros((num_envs, 55), dtype=np.float32)

        self.reset()

    # ---------------- 课程 ----------------
    def _sample_push_angle(self) -> float:
        """推力方向采样：一部分均匀随机，一部分集中到近期失败率最高的方向。

        依据 v1-v15 批量评测，各版本失败集中在少数方向。把课程预算按 24 个
        方向桶的近期摔倒率加权，投向最弱方向；Laplace 平滑保证冷门桶仍被采样。
        v26 用不对称扇区：前向扇区占比高（保护向前卸力迈步），后向只留少量
        样本防止遗忘，其余预算仍按失败率加权投向薄弱方向。
        """
        cfg = self.cfg
        roll = self.rng.random()
        if roll < cfg.fwd_sector_prob + cfg.bwd_sector_prob:
            center = 0.0 if roll < cfg.fwd_sector_prob else np.pi
            half = np.radians(cfg.sector_halfwidth_deg)
            return float((center + self.rng.uniform(-half, half)) % (2 * np.pi))
        if (self.rng.random() >= cfg.weak_dir_prob
                or self.push_try_hist.sum() < 24
                or self.stage()[0] < cfg.push_mag_min_stage):
            return float(self.rng.uniform(0, 2 * np.pi))
        ratio = (self.push_fail_hist + 1.0) / (self.push_try_hist + 2.0)
        k = max(1, min(int(cfg.weak_dir_topk), len(ratio)))
        idx = np.argsort(-ratio)[:k]
        w = ratio[idx] / ratio[idx].sum()
        pick = int(self.rng.choice(idx, p=w))
        half = np.pi / 24.0
        return float((pick + 0.5) * 2 * np.pi / 24.0 + self.rng.uniform(-half, half))

    @staticmethod
    def _bucket(ang: float) -> int:
        return int((ang % (2 * np.pi)) / (2 * np.pi) * 24) % 24

    def _push_scale_for_angle(self, ang: float) -> float:
        """v47: 恢复后向折减（前向 1.0 / 后向 0.90 ≈0.63）—— 与 v45 的配比一致，
        这样 v47 与 v45 的差异只剩"势函数塑形"这一项，可做干净对照。"""
        return 0.90 if np.cos(ang) < 0.0 else 1.0

    def stage(self):
        it_eq = self.total_steps / (NUM_STEPS_PER_ENV * max(self.num_envs, 1))
        row = CURRICULUM[0]
        for r in CURRICULUM:
            if it_eq >= r[0]:
                row = r
        return row

    # ---------------- 指令采样 ----------------
    def _sample_commands(self, ids: np.ndarray):
        if self.cfg.task_mode in ("stand", "push_recovery", "jump"):
            self.commands[ids] = 0.0
            return
        _, _, vx_range, vy_max, wz_max, zero_prob, _ = self.stage()
        n = len(ids)
        zero = self.rng.random(n) < zero_prob
        vx = self.rng.uniform(vx_range[0], vx_range[1], n)
        vy = self.rng.uniform(-vy_max, vy_max, n)
        wz = self.rng.uniform(-wz_max, wz_max, n)
        vx[zero] = 0.0
        vy[zero] = 0.0
        wz[zero] = 0.0
        self.commands[ids] = np.stack([vx, vy, wz], axis=1).astype(np.float32)

    # ---------------- 域随机化（恢复-再-施加） ----------------
    def _randomize(self, ids: np.ndarray):
        cfg = self.cfg
        for i in ids:
            m = self.models[i]
            m.body_mass[:] = self._body_mass0 * self.rng.uniform(*cfg.mass_scale_range, self._body_mass0.shape)
            m.dof_frictionloss[:] = self._frictionloss0 * self.rng.uniform(*cfg.friction_damping_range, self._frictionloss0.shape)
            m.dof_damping[:] = self._damping0 * self.rng.uniform(*cfg.friction_damping_range, self._damping0.shape)
            m.dof_armature[:] = self._armature0 * self.rng.uniform(*cfg.armature_range, self._armature0.shape)
        self.encoder_bias[ids] = self.rng.uniform(
            -cfg.encoder_bias_range, cfg.encoder_bias_range, (len(ids), 14)).astype(np.float32)
        self.action_delay[ids] = self.rng.integers(0, cfg.action_delay_max + 1, len(ids))
        self.push_timer[ids] = self.rng.uniform(*cfg.push_interval_s, len(ids)).astype(np.float32)
        self.push_active[ids] = 0
        self.push_had[ids] = False
        self.push_react[ids] = 0.0

    # ---------------- 重置 ----------------
    def _reset_envs(self, ids: np.ndarray):
        cfg = self.cfg
        self._randomize(ids)
        self._sample_commands(ids)
        for i in ids:
            d = self.datas[i]
            d.qpos[:] = self.home_qpos
            d.qvel[:] = 0.0
            # 关节噪声 + 基座小幅倾斜
            d.qpos[7:] += self.rng.uniform(-cfg.reset_joint_noise, cfg.reset_joint_noise, 14)
            tilt = np.radians(cfg.reset_base_tilt_deg)
            ax = self.rng.uniform(-1, 1, 3)
            norm = np.linalg.norm(ax)
            if norm > 1e-6:
                ax /= norm
                ang = self.rng.uniform(0, tilt)
                s = np.sin(ang / 2)
                d.qpos[3:7] = np.array([np.cos(ang / 2), *(ax * s)])
            d.qpos[2] += self.rng.uniform(0.0, 0.005)
            d.ctrl[:] = self.home_ctrl
            mujoco.mj_forward(self.models[i], d)
            self.base_xy_start[i] = d.qpos[:2]
            self.foot_air_time[i] = 0.0
            self.foot_contact_prev[i] = True
            self.foot_z_lift[i, 0] = d.site_xpos[self._foot_site_ids[0], 2]
            self.foot_z_lift[i, 1] = d.site_xpos[self._foot_site_ids[1], 2]
            self.foot_xy_prev[i, 0] = d.site_xpos[self._foot_site_ids[0], :2]
            self.foot_xy_prev[i, 1] = d.site_xpos[self._foot_site_ids[1], :2]
        self.prev_ctrl[ids] = self.home_ctrl
        self.desired_queue[ids] = self.home_ctrl
        self.last_action[ids] = 0.0
        self.episode_length_buf[ids] = 0
        self.reward_episode_sum[ids] = 0.0
        self._jump_phase[ids] = 0
        self._jump_peak[ids] = cfg.STAND_Z
        self._jump_flight[ids] = 0
        self._base_z_maxseen[ids] = cfg.STAND_Z
        self._foot_both_air_prev[ids] = False
        self._jump_charged[ids] = False

    def reset(self) -> TensorDict:
        self._reset_envs(np.arange(self.num_envs))
        self._gather_and_compute_obs()
        return self.get_observations()

    # ---------------- 观测 ----------------
    def _gather_and_compute_obs(self):
        cfg = self.cfg
        n = self.num_envs
        qpos = np.stack([d.qpos for d in self.datas])          # (N,21)
        qvel = np.stack([d.qvel for d in self.datas])          # (N,20)
        sens = np.stack([d.sensordata for d in self.datas])    # (N,nsens)

        quat = qpos[:, 3:7]
        rot = quat_to_rot(quat)                                 # body->world
        grav_world = np.array([0.0, 0.0, -1.0])
        proj_gravity = np.einsum("nij,j->ni", rot.transpose(0, 2, 1), grav_world)

        omega_body = sens[:, self.gyro_adr:self.gyro_adr + 3]
        v_body = sens[:, self.linvel_adr:self.linvel_adr + 3]
        joint_pos = qpos[:, 7:]
        joint_vel = qvel[:, 6:]
        base_z = qpos[:, 2]

        # 噪声（传感器视图）
        jp = joint_pos + self.encoder_bias + self.rng.normal(0, cfg.noise_joint_pos, joint_pos.shape)
        jv = joint_vel + self.rng.normal(0, cfg.noise_joint_vel, joint_vel.shape)
        om = omega_body + self.rng.normal(0, cfg.noise_gyro, omega_body.shape)
        pg = proj_gravity + self.rng.normal(0, cfg.noise_gravity, proj_gravity.shape)

        cmd = self.commands
        pol = np.concatenate([
            pg.astype(np.float32),
            om.astype(np.float32),
            cmd.astype(np.float32),
            (jp - self.home_ctrl).astype(np.float32),
            jv.astype(np.float32),
            self.last_action.astype(np.float32),
        ], axis=1)
        assert pol.shape[1] == 51, pol.shape

        cri = np.concatenate([
            pol,
            v_body.astype(np.float32),
            (base_z - cfg.base_height_target).astype(np.float32)[:, None],
        ], axis=1)
        assert cri.shape[1] == 55, cri.shape

        self._obs_policy = pol
        self._obs_critic = cri

        # 奖励计算需要的原始量（无噪声）
        self._v_body = v_body
        self._omega_body = omega_body
        self._proj_gravity = proj_gravity
        self._base_z = base_z
        self._joint_pos = joint_pos
        self._joint_vel = joint_vel

    def get_observations(self) -> TensorDict:
        return TensorDict({
            "policy": torch.from_numpy(self._obs_policy).to(self.device),
            "critic": torch.from_numpy(self._obs_critic).to(self.device),
        }, batch_size=[self.num_envs])

    # ---------------- 步 ----------------
    def step(self, actions: torch.Tensor):
        cfg = self.cfg
        a = actions.detach().cpu().numpy().astype(np.float32)
        a = np.clip(a, -1.0, 1.0)

        desired = self.home_ctrl + cfg.action_scale * a
        use_delayed = (self.action_delay > 0)[:, None]
        applied = np.where(use_delayed, self.desired_queue, desired)
        self.desired_queue = desired.copy()

        max_delta = cfg.max_motor_velocity * cfg.ctrl_dt
        ctrl = self.prev_ctrl + np.clip(applied - self.prev_ctrl, -max_delta, max_delta)
        self.prev_ctrl = ctrl

        for i in range(self.num_envs):
            self.datas[i].ctrl[:] = ctrl[i]
        for _ in range(cfg.decimation):
            for i in range(self.num_envs):
                mujoco.mj_step(self.models[i], self.datas[i])

        # ---- v2 步态指标：脚底接触 / 摆动抬脚 ----
        contact = np.zeros((self.num_envs, 2), dtype=bool)
        foot_z = np.zeros((self.num_envs, 2), dtype=np.float64)
        foot_xy = np.zeros((self.num_envs, 2, 2), dtype=np.float64)
        for i in range(self.num_envs):
            d = self.datas[i]
            for ci in range(d.ncon):
                ct = d.contact[ci]
                if ct.geom1 in self._foot_geom_idx:
                    contact[i, self._foot_geom_idx[ct.geom1]] = True
                if ct.geom2 in self._foot_geom_idx:
                    contact[i, self._foot_geom_idx[ct.geom2]] = True
            foot_z[i, 0] = d.site_xpos[self._foot_site_ids[0], 2]
            foot_z[i, 1] = d.site_xpos[self._foot_site_ids[1], 2]
            foot_xy[i, 0] = d.site_xpos[self._foot_site_ids[0], :2]
            foot_xy[i, 1] = d.site_xpos[self._foot_site_ids[1], :2]

        cmd_speed = np.abs(cmd0 := self.commands[:, 0]) + np.abs(self.commands[:, 1]) + np.abs(self.commands[:, 2])
        active = cmd_speed > 0.01
        airborne = ~contact
        shortfall = np.maximum(0.0, (self.foot_z_lift + cfg.foot_swing_target) - foot_z)
        swing_cost = (airborne * shortfall ** 2).sum(axis=1)
        touchdown = (~self.foot_contact_prev) & contact
        air_reward = (touchdown * np.clip(self.foot_air_time - cfg.air_time_min, 0.0,
                                          cfg.air_time_max - cfg.air_time_min)).sum(axis=1) * active
        liftoff = self.foot_contact_prev & airborne
        self.foot_z_lift[liftoff] = foot_z[liftoff]
        foot_v_xy = (foot_xy - self.foot_xy_prev) / cfg.ctrl_dt
        self.foot_xy_prev = foot_xy
        low = foot_z < cfg.foot_clearance_z
        clearance_cost = (low * (foot_v_xy ** 2).sum(axis=2)).sum(axis=1) * active
        self.foot_air_time = np.where(airborne, self.foot_air_time + cfg.ctrl_dt, 0.0)
        self.push_react = np.maximum(0.0, self.push_react - cfg.ctrl_dt)
        self.foot_contact_prev = contact

        # 推力扰动（microduck 风格瞬时冲量）：大小随课程 stage()[6] 上限增长，
        # 单控制步一次性给速度冲量、随机方向，符合 9/6 训练时的语义。
        push_mag = self.stage()[6] * self.push_scale
        if push_mag > 0:
            self.push_timer -= cfg.ctrl_dt
            hit = np.where(self.push_timer <= 0)[0]
            for i in hit:
                ang = self._sample_push_angle()
                self.push_ang[i] = ang
                self.push_ang_log[i] = ang
                self.push_dv[i] = push_mag * self._push_scale_for_angle(ang)
                self.push_had[i] = True
                self.push_try_hist[self._bucket(ang)] += 1.0
                self.push_timer[i] = self.rng.uniform(*cfg.push_interval_s)
            for i in np.where(self.push_dv > 0)[0]:
                step = self.push_dv[i]
                pa = self.push_ang[i]
                self.datas[i].qvel[0] += step * np.cos(pa)
                self.datas[i].qvel[1] += step * np.sin(pa)
                self.push_base_xy[i] = self.datas[i].qpos[:2]   # v37: 记录被推瞬间的基座位置
                self.push_dv[i] = 0.0
                self.push_react[i] = cfg.push_react_window_s

        self.episode_length_buf += 1
        self.total_steps += self.num_envs
        self._gather_and_compute_obs()

        # ---- 奖励（统一约定：惩罚=代价>=0 × 负权重）----
        cmd = self.commands
        vb = self._v_body
        om = self._omega_body
        pg = self._proj_gravity

        upright = np.exp(-(pg[:, 0] ** 2 + pg[:, 1] ** 2) / cfg.upright_std2)
        stillness = np.exp(-(vb[:, 0] ** 2 + vb[:, 1] ** 2) / cfg.stillness_std2) * (~active)

        err_lin = (vb[:, 0] - cmd[:, 0]) ** 2 + (vb[:, 1] - cmd[:, 1]) ** 2
        track_lin = np.exp(-err_lin / cfg.sigma_track_lin ** 2)
        err_ang = (om[:, 2] - cmd[:, 2]) ** 2
        track_ang = np.exp(-err_ang / cfg.sigma_track_ang ** 2)

        low_height = np.maximum(0.0, cfg.base_height_low - self._base_z)
        high_height = np.maximum(0.0, self._base_z - cfg.base_height_high)
        cost_height = (low_height ** 2) * abs(cfg.w_base_height_low) + (high_height ** 2) * abs(cfg.w_base_height_high)
        cost_orient = pg[:, 0] ** 2 + pg[:, 1] ** 2
        cost_vz = vb[:, 2] ** 2
        cost_angxy = om[:, 0] ** 2 + om[:, 1] ** 2
        q = self._joint_pos
        cost_limits = (np.maximum(0.0, self.soft_low - q) ** 2 +
                       np.maximum(0.0, q - self.soft_up) ** 2).sum(axis=1)
        cost_action_rate = ((a - self.last_action) ** 2).sum(axis=1)

        w_action_rate = self.stage()[1]
        if cfg.task_mode in ("stand", "push_recovery"):
            # ── microduck standup 配方 ──
            z = self._base_z
            tilt2 = cost_orient
            tilt = np.sqrt(tilt2)
            # 高度：宽/窄两层高斯（固定目标 STAND_Z，非区间惩罚）
            h_wide = np.exp(-(z - cfg.STAND_Z) ** 2 / (2 * cfg.height_std ** 2))
            h_sharp = np.exp(-(z - cfg.STAND_Z) ** 2 / (2 * cfg.height_std_sharp ** 2))
            h_l1 = np.abs(z - cfg.STAND_Z)
            # 上升速度奖励：z 在 STAND_Z 下方时奖励 vz>0
            vz_up = np.maximum(vb[:, 2], 0.0) * (z < cfg.com_upward_max_height)
            # 直立：线性 cos(tilt) 粗牵引
            up_lin = np.cos(np.clip(tilt, 0.0, np.pi / 2))
            # 高度门控的锐化直立高斯（防“蹲低+竖直”作弊）
            gate_h = np.clip((z - cfg.base_height_low) / (cfg.STAND_Z - cfg.base_height_low), 0.0, 1.0)
            up_sharp = np.exp(-tilt2 / (2 * cfg.upright_sharp_std ** 2)) * gate_h
            # 姿态匹配：腿关节贴近 HOME
            q_leg = q[:, LEG_JOINTS]
            home_leg = self.home_ctrl[LEG_JOINTS]
            leg_err = np.sqrt(((q_leg - home_leg) ** 2).mean(axis=1))
            pose_stand = np.exp(-(leg_err ** 2) / (2 * cfg.pose_std ** 2))
            pose_l1 = leg_err
            q_head = q[:, 5:9]
            head_target = np.asarray(cfg.stand_head_target, dtype=np.float64)
            head_weights = np.asarray(cfg.stand_head_joint_weights, dtype=np.float64)
            head_delta = q_head - head_target
            head_err = np.sqrt((head_weights * head_delta ** 2).sum(axis=1) /
                               head_weights.sum())
            head_pose_stand = np.exp(-(head_err ** 2) / (2 * cfg.head_pose_std ** 2))
            head_limit_cost = np.sum(
                np.maximum(self.soft_low[5:9] - q_head, 0.0) ** 2
                + np.maximum(q_head - self.soft_up[5:9], 0.0) ** 2,
                axis=1,
            )
            # 组合分：高度×直立×姿态
            comp = (np.exp(-(z - cfg.STAND_Z) ** 2 / (2 * cfg.height_std ** 2)) *
                    np.exp(-tilt2 / (2 * cfg.upright_sharp_std ** 2)) *
                    np.exp(-(leg_err ** 2) / (2 * cfg.stand_pose_std ** 2)))
            # v32 姿态门控（v27 机制）：被推反应窗口内放开姿态约束。
            # 窗口里正是需要抬脚换重心的时候，还罚姿态会跟卸力迈步互斥；
            # 窗口外姿态照常约束 → 站姿不受影响。
            react = (self.push_react > 0.0).astype(np.float64)
            still_gate = 1.0 - react
            reward_rate = (
                cfg.w_alive
                + cfg.w_height_gauss * h_wide
                + cfg.w_height_gauss_sharp * h_sharp
                - cfg.w_height_l1 * h_l1
                + cfg.w_com_upward * vz_up
                + cfg.w_upright_linear * up_lin
                + cfg.w_upright_sharp * up_sharp
                + cfg.w_standing_composite * comp
                + still_gate * (cfg.w_pose_stand * pose_stand
                                - cfg.w_pose_l1 * pose_l1
                                + cfg.w_head_pose_stand * head_pose_stand
                                - cfg.w_head_pose_l1 * head_err)
                + cfg.w_head_limit * head_limit_cost
                + cfg.w_orientation * cost_orient
                + cfg.w_lin_vel_z * cost_vz
                + cfg.w_ang_vel_xy * cost_angxy
                + cfg.w_joint_limits * cost_limits
                + w_action_rate * cost_action_rate
            )
            if cfg.task_mode == "push_recovery":
                # 直立进度势形：Δ(−cost_orient)，越直立越大，越倒越负
                up_progress = self._tilt_cos_prev - cost_orient
                tilt_deg = np.degrees(np.sqrt(cost_orient))
                in_recovery = tilt_deg > cfg.recovery_tilt_deg
                single_foot = np.max(self.foot_air_time, axis=1)
                both_feet_air = np.all(airborne, axis=1)
                recover_step = in_recovery * (~both_feet_air) * np.clip(
                    single_foot / cfg.air_time_max, 0.0, 1.0)
                # 卸力行走：机体 world 速度朝推力方向的分量越大越给分
                vel_w = np.stack([d.qvel[0:2] for d in self.datas])      # (N,2) world
                v_along = (np.cos(self.push_ang) * vel_w[:, 0]
                           + np.sin(self.push_ang) * vel_w[:, 1])
                recover_walk = in_recovery * np.clip(
                    v_along / cfg.recover_walk_vmax, 0.0, 1.0)
                gait_shape = (
                    cfg.w_air_time * air_reward
                    + cfg.w_foot_swing * swing_cost
                    + cfg.w_foot_clearance * clearance_cost
                )
                # 防"被推就走"退化：直立且低速时惩罚水平漂移与速度。
                # v8→v16 对照显示，一旦允许把推力转成行走，策略会连无推力时
                # 也踏步漂移（10s 抬脚 197 次、漂移 0.34m）。该项把"原地站住"
                # 重新设为最优解，同时用位移上限避免僵硬摔倒。
                base_xy = np.stack([d.qpos[0:2] for d in self.datas])
                cost_drift = np.minimum(
                    ((base_xy - self.base_xy_start) ** 2).sum(axis=1),
                    cfg.play_drift_cap ** 2)
                play_still = (tilt_deg < cfg.play_still_tilt_deg).astype(np.float64)
                # v26 推力门控卸力迈步：只在刚被推的短窗口内计分。
                #   yield_step = 单脚离地（换脚承重）
                #   yield_walk = 顺着推力方向让位（不硬顶）
                # 两者都要求机体还没倒（tilt 有上限），防止"直接躺平"刷分。
                # still_gate 已在上面姿态项处算好（= 1 - react），这里直接复用（v32）。
                yield_ok = react * (~both_feet_air) * (tilt_deg < cfg.yield_tilt_max_deg)
                # v46 势函数门控：本步倾角不能变大才给卸力分（microduck 发现 1）。
                # up_progress = prev_tilt² − cur_tilt² > 0 表示正在被扶正。
                if cfg.yield_recover_only:
                    yield_ok = yield_ok * (up_progress > cfg.recover_gate_eps).astype(np.float64)
                yield_step = yield_ok * np.clip(single_foot / cfg.air_time_max, 0.0, 1.0)
                # v38: 窗口内单脚悬空超时 → 逐脚累加惩罚（脚不该停在半空）
                long_air = react[:, None] * np.clip(self.foot_air_time - cfg.react_flight_free_s,
                                                   0.0, None)
                cost_react_flight = long_air.sum(axis=1)
                # v33 方向门控的让位走：只在"推背"（推力含 -x 分量）时奖励顺着
                # 让位 —— 往后让位会摔，往前让位才是卸力；前推只留原地抬脚卸力，
                # 避免 v32 那种"让步↔硬顶"双稳态摆动（前沿 F=0 交替出现）。
                back_gate = (np.cos(self.push_ang) < 0.0).astype(np.float64)
                yield_walk = yield_ok * back_gate * np.clip(v_along / cfg.recover_walk_vmax, 0.0, 1.0)
                # v37 垫步接住：窗口内刚落地的脚，若踩在推力方向一侧（相对被推瞬间
                # 的基座），按偏移量给分。方向无关 —— 前推往前垫、后推往后垫同理。
                push_dir = np.stack([np.cos(self.push_ang), np.sin(self.push_ang)], axis=1)
                foot_along = ((foot_xy - self.push_base_xy[:, None, :]) *
                              push_dir[:, None, :]).sum(axis=2)          # (N,2) 各脚沿推力方向的偏移
                catch_step = (touchdown.astype(np.float64) * react[:, None] *
                              np.clip(foot_along / cfg.catch_step_ref, 0.0, 1.0)).sum(axis=1)
                yield_shape = (cfg.w_yield_step * yield_step
                               + cfg.w_yield_walk * yield_walk
                               + cfg.w_catch_step * catch_step)
                # v32 空闲踏步惩罚（v28 机制）：推力窗口外脚不该离地。
                # 刻意不加 tilt 门控 —— play_still 的门控(tilt<6°)会被"踏步把常态
                # 倾角抬高"利用；这里只认 within-window 与否，且权重温和防止死扛。
                idle_air = still_gate[:, None] * airborne.astype(np.float64)
                cost_idle_lift = idle_air.sum(axis=1)
                # v46: 窗口内把"正在被扶正"的信号放大（势函数导向），
                # 并对"停在倾斜态"持续扣分 —— 两者一起把偷分路径堵死。
                cost_tilt_over = react * np.clip(tilt_deg - cfg.tilt_over_deg, 0.0, None)
                reward_rate = (reward_rate
                               + cfg.w_upright_progress * (1.0 + cfg.react_up_boost * react) * up_progress
                               + cfg.w_react_upright * react * up_progress
                               + cfg.w_tilt_over * cost_tilt_over
                               + cfg.w_recovery_step * recover_step
                               + cfg.w_recover_walk * recover_walk
                               + yield_shape
                               + cfg.w_play_still * still_gate * play_still * (vb[:, 0] ** 2 + vb[:, 1] ** 2)
                               + cfg.w_play_drift * still_gate * cost_drift
                               + cfg.w_idle_lift * cost_idle_lift
                               + cfg.w_react_flight * cost_react_flight
                               + gait_shape)
                self._tilt_cos_prev = cost_orient.copy()
        elif cfg.task_mode == "jump":
            # ── 干净的跳跃奖励（不再叠在站姿奖励上）──
            # 移除会阻止跳跃的站姿惩罚：height L1(−7.5|z−0.15|) 拦下蹲、lin_vel_z(−2vz²) 拦起身，
            # 这两项正是前几轮模型只会站着刷分、永不离地的元凶。
            # 核心信条："蹲下蓄力、起身一定要快" —— rise(向上速度) 是最强奖励项。
            z = self._base_z
            vz = vb[:, 2]
            tilt = np.sqrt(cost_orient)
            both_air = np.all(airborne, axis=1)
            both_ground = np.all(contact, axis=1)
            up_lin = np.cos(np.clip(tilt, 0.0, np.pi / 2))
            # 蓄力深度：相对 STAND_Z 下探（0..1，满蹲=-0.04m）
            crouch_depth = np.clip((cfg.STAND_Z - z) / cfg.jump_crouch_depth, 0.0, 1.0)
            # 下蹲蓄力：深蹲+下蹲速度 → 为快速起身储能（pre-load）
            crouch = crouch_depth * np.clip(-vz, 0.0, None)
            # 起身爆发：连续单调奖励，避免 0.85m/s sigmoid 造成梯度荒漠。
            # 时序门控：必须"先蹲下蓄力(双膝屈曲到位)"起身才给大分 —— 直接编码
            # "膝关节先压缩、再迅速打开"的弹簧时序，杜绝可视化里看到的脚尖硬顶无膝压缩。
            rising = (z < cfg.STAND_Z + 0.04)
            vz_up = np.clip(vz, 0.0, None)
            burst = vz_up / (vz_up + cfg.jump_rise_kref)
            # 双膝相对 HOME 的屈曲深度（0=直立,>=1=满蹲蓄力）
            knee = self._joint_pos[:, [3, 12]]
            knee_flex = np.clip((self.home_qpos[[10, 19]] - knee) / cfg.jump_knee_compress_ref, 0.0, None)
            deep_crouch = (knee_flex.min(axis=1) > cfg.jump_knee_ready_ratio) & (z < cfg.STAND_Z)
            # 充能：真蹲→置位；双脚离地→消耗(一次蹲只发一次身,蹬伸全程都算)
            self._jump_charged = self._jump_charged | deep_crouch
            rise = burst * rising * (~both_air) * self._jump_charged
            # 已充能后快速打开膝盖（蹬伸速度）→ 直接把"蹲下后迅速蹬开"写进奖励
            knee_ext = np.clip(self._joint_vel[:, [3, 12]], 0.0, None).mean(axis=1)
            knee_extend = self._jump_charged * np.clip(knee_ext / cfg.jump_knee_extend_ref, 0.0, 1.0) * (vz_up > 0.0)
            self._jump_charged = self._jump_charged & ~both_air
            # 腾空：双脚离地时的基座离地高度
            flight = both_air.astype(np.float64) * np.clip(z - cfg.STAND_Z, 0.0, 0.08)
            # 顶到最高：本跳全程最高点（先并入当前 z）
            self._base_z_maxseen = np.maximum(self._base_z_maxseen, z)
            apex = both_air.astype(np.float64) * np.clip(self._base_z_maxseen - cfg.STAND_Z, 0.0, 0.10)
            # 落地：上一控制步双脚离地 → 现在双脚着地。只有真正离地超过阈值才算一次真跳。
            landed = both_ground & self._foot_both_air_prev & (self._base_z_maxseen > cfg.STAND_Z + 0.005)
            jump_apex = self._base_z_maxseen - cfg.STAND_Z
            jump_ok = landed & (jump_apex > cfg.jump_min_cycle_apex)
            self._foot_both_air_prev = both_air.copy()
            # 一次完整真跳的结算：按最高点比例发大分（越蹦越高越赚），小跳(<2cm)给零
            landing = jump_ok * np.clip(jump_apex, 0.0, 0.10) * cfg.jump_w_cycle
            landing += landed.astype(np.float64) * np.exp(-np.abs(vz) / 0.5)  # 软落地闭环
            # 落地后把最高点清零，备下一次跳跃
            self._base_z_maxseen = np.where(landed, z, self._base_z_maxseen)
            self._jump_cycles += jump_ok.astype(np.int64)
            reward_rate = (
                cfg.w_alive
                + cfg.w_orientation * cost_orient
                + cfg.w_ang_vel_xy * cost_angxy
                + cfg.w_joint_limits * cost_limits
                + w_action_rate * cost_action_rate
                + cfg.w_upright_linear * up_lin
                + cfg.jump_w_crouch * crouch
                + cfg.jump_w_takeoff * rise
                + cfg.jump_w_knee_extend * knee_extend
                + cfg.jump_w_flight * flight
                + cfg.jump_w_apex * apex
                + cfg.jump_w_landing * landing
                + cfg.jump_w_impact * (landed * vz ** 2)
            )
        else:
            track_weight = 1.0 if cfg.task_mode == "velocity" else 0.0
            reward_rate = (
                track_weight * (cfg.w_track_lin * track_lin + cfg.w_track_ang * track_ang)
                + cfg.w_alive
                - cost_height
                + cfg.w_orientation * cost_orient
                + cfg.w_lin_vel_z * cost_vz
                + cfg.w_ang_vel_xy * cost_angxy
                + cfg.w_joint_limits * cost_limits
                + w_action_rate * cost_action_rate
                + cfg.w_air_time * air_reward
                + cfg.w_foot_swing * swing_cost
                + cfg.w_upright * upright
                + cfg.w_stillness * stillness
                + cfg.w_foot_clearance * clearance_cost
            )
        rewards = reward_rate * cfg.ctrl_dt

        # ---- 终止 ----
        fallen = (cost_orient > cfg.fall_tilt_cost) | (self._base_z < cfg.min_height) | ((self._base_z > cfg.max_height) & (cfg.task_mode != "jump"))
        timeout = self.episode_length_buf >= cfg.episode_length
        dones = fallen | timeout
        rewards[fallen] += cfg.w_fall

        self.reward_episode_sum += rewards
        self.last_action = a.copy()

        # 统计 + 自动重置
        done_ids = np.where(dones)[0]
        if cfg.task_mode == "push_recovery" and len(done_ids):
            for i in done_ids:
                if fallen[i] and self.push_had[i]:
                    self.push_fail_hist[self._bucket(self.push_ang_log[i])] += 1.0
        if cfg.task_mode == "push_recovery" and len(done_ids):
            # 新回合从直立起点开始，避免跨回合误算“恢复进度”
            self._tilt_cos_prev[done_ids] = 0.0
        if len(done_ids):
            self._finished_r.extend(self.reward_episode_sum[done_ids].tolist())
            self._finished_l.extend(self.episode_length_buf[done_ids].tolist())
            self._reset_envs(done_ids)
            self._gather_and_compute_obs()  # 重置后刷新观测

        # ---- extras ----
        extras: dict = {"time_outs": torch.from_numpy(timeout.astype(bool)).to(self.device)}
        ep: dict = {
            "Rewards/track_lin_vel": torch.full((self.num_envs,), cfg.w_track_lin * cfg.ctrl_dt) * torch.from_numpy(track_lin).float(),
            "Rewards/track_ang_vel": torch.from_numpy((cfg.w_track_ang * track_ang) * cfg.ctrl_dt).float(),
            "Rewards/alive": torch.full((self.num_envs,), cfg.w_alive * cfg.ctrl_dt),
            "Rewards/base_height": torch.from_numpy((-cost_height * cfg.ctrl_dt)).float(),
            "Rewards/orientation": torch.from_numpy((cfg.w_orientation * cost_orient) * cfg.ctrl_dt).float(),
            "Rewards/lin_vel_z": torch.from_numpy((cfg.w_lin_vel_z * cost_vz) * cfg.ctrl_dt).float(),
            "Rewards/ang_vel_xy": torch.from_numpy((cfg.w_ang_vel_xy * cost_angxy) * cfg.ctrl_dt).float(),
            "Rewards/joint_limits": torch.from_numpy((cfg.w_joint_limits * cost_limits) * cfg.ctrl_dt).float(),
            "Rewards/action_rate": torch.from_numpy((w_action_rate * cost_action_rate) * cfg.ctrl_dt).float(),
            "Rewards/fall_penalty": torch.from_numpy(np.where(fallen, cfg.w_fall, 0.0)).float(),
            "Rewards/air_time": torch.from_numpy((cfg.w_air_time * air_reward) * cfg.ctrl_dt).float(),
            "Rewards/foot_swing": torch.from_numpy((cfg.w_foot_swing * swing_cost) * cfg.ctrl_dt).float(),
            "Rewards/upright": torch.from_numpy((cfg.w_upright * upright) * cfg.ctrl_dt).float(),
            "Rewards/stillness": torch.from_numpy((cfg.w_stillness * stillness) * cfg.ctrl_dt).float(),
            "Rewards/foot_clearance": torch.from_numpy((cfg.w_foot_clearance * clearance_cost) * cfg.ctrl_dt).float(),
            "Cmd/vx": torch.from_numpy(cmd[:, 0]).float(),
            "State/base_z": torch.from_numpy(self._base_z).float(),
            "State/tilt_cost": torch.from_numpy(cost_orient).float(),
        }
        if cfg.task_mode == "push_recovery":
            ep["Rewards/recover_walk"] = torch.from_numpy((cfg.w_recover_walk * recover_walk) * cfg.ctrl_dt).float()
            ep["Rewards/play_still"] = torch.from_numpy((cfg.w_play_still * play_still * (vb[:, 0] ** 2 + vb[:, 1] ** 2)) * cfg.ctrl_dt).float()
            ep["Rewards/play_drift"] = torch.from_numpy((cfg.w_play_drift * cost_drift) * cfg.ctrl_dt).float()
            ep["Rewards/yield_step"] = torch.from_numpy((cfg.w_yield_step * yield_step) * cfg.ctrl_dt).float()
            ep["Rewards/yield_walk"] = torch.from_numpy((cfg.w_yield_walk * yield_walk) * cfg.ctrl_dt).float()
            ep["Rewards/catch_step"] = torch.from_numpy((cfg.w_catch_step * catch_step) * cfg.ctrl_dt).float()
            ep["Rewards/react_flight"] = torch.from_numpy((cfg.w_react_flight * cost_react_flight) * cfg.ctrl_dt).float()
            ep["Rewards/tilt_over"] = torch.from_numpy((cfg.w_tilt_over * cost_tilt_over) * cfg.ctrl_dt).float()
            ep["State/up_progress"] = torch.from_numpy(up_progress).float()
            ep["Rewards/idle_lift"] = torch.from_numpy((cfg.w_idle_lift * cost_idle_lift) * cfg.ctrl_dt).float()
            ep["State/leg_err"] = torch.from_numpy(leg_err).float()
            ep["State/react"] = torch.from_numpy(react).float()
        if cfg.task_mode == "stand":
            ep["Rewards/stand_height_wide"] = torch.from_numpy((cfg.w_height_gauss * h_wide) * cfg.ctrl_dt).float()
            ep["Rewards/stand_height_sharp"] = torch.from_numpy((cfg.w_height_gauss_sharp * h_sharp) * cfg.ctrl_dt).float()
            ep["Rewards/stand_height_l1"] = torch.from_numpy((-cfg.w_height_l1 * h_l1) * cfg.ctrl_dt).float()
            ep["Rewards/stand_upward"] = torch.from_numpy((cfg.w_com_upward * vz_up) * cfg.ctrl_dt).float()
            ep["Rewards/stand_upright_lin"] = torch.from_numpy((cfg.w_upright_linear * up_lin) * cfg.ctrl_dt).float()
            ep["Rewards/stand_upright_sharp"] = torch.from_numpy((cfg.w_upright_sharp * up_sharp) * cfg.ctrl_dt).float()
            ep["Rewards/stand_pose"] = torch.from_numpy((cfg.w_pose_stand * pose_stand) * cfg.ctrl_dt).float()
            ep["Rewards/stand_composite"] = torch.from_numpy((cfg.w_standing_composite * comp) * cfg.ctrl_dt).float()
            ep["State/leg_err"] = torch.from_numpy(leg_err).float()
            ep["Rewards/stand_head_pose"] = torch.from_numpy((cfg.w_head_pose_stand * head_pose_stand) * cfg.ctrl_dt).float()
            ep["Rewards/stand_head_l1"] = torch.from_numpy((-cfg.w_head_pose_l1 * head_err) * cfg.ctrl_dt).float()
            ep["State/head_err"] = torch.from_numpy(head_err).float()
        if cfg.task_mode == "jump":
            ep["State/jump_cycles"] = torch.from_numpy(self._jump_cycles.astype(np.float32)).float()
            ep["State/base_z_maxseen"] = torch.from_numpy(self._base_z_maxseen.astype(np.float32)).float()
            ep["Rewards/jump_takeoff"] = torch.from_numpy((cfg.jump_w_takeoff * rise) * cfg.ctrl_dt).float()
            ep["Rewards/jump_flight"] = torch.from_numpy((cfg.jump_w_flight * flight) * cfg.ctrl_dt).float()
            ep["Rewards/jump_apex"] = torch.from_numpy((cfg.jump_w_apex * apex) * cfg.ctrl_dt).float()
        extras["episode"] = ep

        obs = self.get_observations()
        rew_t = torch.from_numpy(rewards.astype(np.float32)).to(self.device)
        done_t = torch.from_numpy(dones.astype(bool)).to(self.device)
        return obs, rew_t, done_t, extras

    # ---------------- 辅助 ----------------
    def pop_episode_stats(self) -> dict:
        if not self._finished_r:
            return {}
        stats = {
            "return_mean": float(np.mean(self._finished_r)),
            "length_mean": float(np.mean(self._finished_l)),
            "count": len(self._finished_r),
        }
        self._finished_r.clear()
        self._finished_l.clear()
        return stats

    def as_dict(self) -> dict:
        d = asdict(self.cfg)
        d["num_envs"] = self.num_envs
        d["curriculum"] = [list(r) for r in CURRICULUM]
        return d
