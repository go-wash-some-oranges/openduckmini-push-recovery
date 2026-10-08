#!/usr/bin/env python3
"""两种推力施加语义下的抗推存活率对比。

mode=train : 冲量在 env.step 末施加（训练/扫描语义），策略下一步即可反应
mode=delay : 冲量在 env.step 前施加（渲染语义），策略要多等一个控制周期

用法:
  python eval_push_modes.py --dir v24 --ckpt <pt> --angles 0 180 --mags 0.60 0.70 \
      --seeds 20 --seed0 6000
"""
from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_mod(vdir: Path):
    spec = importlib.util.spec_from_file_location(
        f"epm_{abs(hash(str(vdir)))}", vdir / "duck_vec_env.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def build_policy(ckpt: str):
    from rsl_rl.modules import ActorCritic

    sample = TensorDict(
        {"policy": torch.zeros(1, 51), "critic": torch.zeros(1, 55)},
        batch_size=[1],
    )
    policy = ActorCritic(
        sample, {"policy": ["policy"], "critic": ["critic"]}, num_actions=14,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128],
        activation="elu", actor_obs_normalization=True,
        critic_obs_normalization=True, init_noise_std=0.5,
        noise_std_type="scalar",
    )
    ckpt_data = torch.load(ckpt, map_location="cpu", weights_only=False)
    policy.load_state_dict(ckpt_data["model_state_dict"])
    policy.eval()
    return policy


def make_env(mod, seed: int, reset_noise: float):
    cfg = dataclasses.replace(
        mod.MicroCfg(task_mode="push_recovery"),
        noise_joint_pos=0.0, noise_joint_vel=0.0, noise_gyro=0.0,
        noise_gravity=0.0, encoder_bias_range=0.0, action_delay_max=0,
        mass_scale_range=(1.0, 1.0), friction_damping_range=(1.0, 1.0),
        armature_range=(1.0, 1.0), reset_joint_noise=reset_noise,
        reset_base_tilt_deg=0.0,
    )
    env = mod.DuckVecEnv(cfg, num_envs=1, seed=seed, device="cpu")
    env.total_steps = 10**9
    env.push_timer[:] = 1e9
    return env


def run_one(env, policy, angle_deg, mag, seed, warmup, horizon, mode):
    with torch.inference_mode():
        env.rng = np.random.default_rng(seed)
        obs = env.reset()
        env.push_timer[:] = 1e9
        for _ in range(warmup):
            obs, *_ = env.step(policy.act_inference(obs))

        ang = np.radians(angle_deg)
        if mode == "train":
            env.push_dv[0] = mag
            env.push_ang[0] = ang
        elif mode == "delay":
            # 渲染语义：冲量在物理积分之前一次性施加，策略要多等一个控制周期
            d = env.datas[0]
            d.qvel[0] += mag * np.cos(ang)
            d.qvel[1] += mag * np.sin(ang)
        alive, t = True, -1
        for t in range(horizon):
            obs, _, done, _ = env.step(policy.act_inference(obs))
            if bool(done[0].item()):
                alive = False
                break
        return alive, t + 1


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".",
                    help="含 duck_vec_env.py 的目录（默认使用仓库根目录的规范环境）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--angles", nargs="+", type=float, default=[0.0, 180.0])
    ap.add_argument("--mags", nargs="+", type=float, default=[0.60, 0.70])
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--seed0", type=int, default=2300)
    ap.add_argument("--warmup", type=int, default=60)
    ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--reset-noise", type=float, default=0.0)
    ap.add_argument("--reuse-env", action="store_true",
                    help="复用同一个 env 实例（会带入 MuJoCo 求解器热启动残留，存活率虚高）")
    ap.add_argument("--modes", nargs="+", default=["train", "delay"],
                    choices=["train", "delay"])
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    vdir = (ROOT / args.dir).resolve()
    mod = load_mod(vdir)
    policy = build_policy(args.ckpt)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))

    env_shared = make_env(mod, args.seed0, args.reset_noise) if args.reuse_env else None

    def episode(angle, mag, seed, mode):
        env = env_shared if env_shared is not None else make_env(mod, seed, args.reset_noise)
        return run_one(env, policy, angle, mag, seed, args.warmup, args.horizon, mode)

    tag = args.tag or Path(args.ckpt).stem
    print(f"{'model':>18} {'mode':>6} {'angle':>6} {'mag':>5} {'surv%':>6}")
    for angle in args.angles:
        for mag in args.mags:
            for mode in args.modes:
                rows = [episode(angle, mag, s, mode) for s in seeds]
                print(f"{tag:>18} {mode:>6} {angle:6.0f} {mag:5.2f} "
                      f"{100 * np.mean([r[0] for r in rows]):6.0f}")


if __name__ == "__main__":
    main()
