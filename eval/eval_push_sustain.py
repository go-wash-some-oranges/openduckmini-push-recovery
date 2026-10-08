#!/usr/bin/env python3
"""持续推力语义下的抗推评测 —— 对齐 push_viewer 里"回车"的操作手感。

push_viewer 的持续推力：把总冲量 mag 摊到 --push-duration 秒（默认 0.3s），
每个控制步给 mag/(dur/dt) 的速度增量。训练里是**瞬时不摊销**，所以这个脚本
测的是"用户真实手推"下的存活率，与 eval_push_modes 的瞬时语义互补。

用法:
  python eval_push_sustain.py --dir v38 --ckpt <pt> --angles 180 --mags 0.70 0.80 0.90 \
      --dur 0.3 --seeds 60 --seed0 6000 --reset-noise 0.03
"""
from __future__ import annotations
import argparse, dataclasses, importlib.util, sys
from pathlib import Path
import numpy as np, torch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def load_mod(vdir: Path):
    spec = importlib.util.spec_from_file_location(f"eps_{abs(hash(str(vdir)))}", vdir / "duck_vec_env.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod); return mod


def build_policy(ckpt: str):
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    sample = TensorDict({"policy": torch.zeros(1, 51), "critic": torch.zeros(1, 55)}, batch_size=[1])
    pol = ActorCritic(sample, {"policy": ["policy"], "critic": ["critic"]}, num_actions=14,
        actor_hidden_dims=[512, 256, 128], critic_hidden_dims=[512, 256, 128], activation="elu",
        actor_obs_normalization=True, critic_obs_normalization=True, init_noise_std=0.5, noise_std_type="scalar")
    pol.load_state_dict(torch.load(ckpt, map_location="cpu", weights_only=False)["model_state_dict"])
    pol.eval(); return pol


def make_env(mod, seed, reset_noise):
    cfg = dataclasses.replace(mod.MicroCfg(task_mode="push_recovery"),
        noise_joint_pos=0.0, noise_joint_vel=0.0, noise_gyro=0.0, noise_gravity=0.0,
        encoder_bias_range=0.0, action_delay_max=0, mass_scale_range=(1.0, 1.0),
        friction_damping_range=(1.0, 1.0), armature_range=(1.0, 1.0),
        reset_joint_noise=reset_noise, reset_base_tilt_deg=0.0)
    # v43 及以后：env 自带 sustain 机制时置 1，避免与脚本的分摊叠加
    if hasattr(cfg, "sustain_push_steps"):
        cfg = dataclasses.replace(cfg, sustain_push_steps=1)
    env = mod.DuckVecEnv(cfg, num_envs=1, seed=seed, device="cpu")
    env.total_steps = 10**9; env.push_timer[:] = 1e9
    return env


def run_one(env, policy, angle_deg, mag, dur, seed, warmup, horizon):
    with torch.inference_mode():
        env.rng = np.random.default_rng(seed)
        obs = env.reset(); env.push_timer[:] = 1e9
        if hasattr(env, "push_dv_left"):
            env.push_dv_left[:] = 0
        for _ in range(warmup):
            obs, *_ = env.step(policy.act_inference(obs))
        ang = np.radians(angle_deg)
        n = max(1, int(round(dur / env.cfg.ctrl_dt)))
        step_dv = mag / n
        d = env.datas[0]
        alive, peak = True, 0.0
        for t in range(horizon):
            if t < n:                       # 持续推力：前 n 步每步加 step_dv
                d.qvel[0] += step_dv * np.cos(ang)
                d.qvel[1] += step_dv * np.sin(ang)
            obs, _, done, _ = env.step(policy.act_inference(obs))
            pg = env._proj_gravity[0]
            peak = max(peak, float(np.degrees(np.hypot(pg[0], pg[1]))))
            if bool(done[0].item()):
                alive = False; break
        return alive, peak


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=".",
                    help="含 duck_vec_env.py 的目录（默认使用仓库根目录的规范环境）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--angles", nargs="+", type=float, default=[180.0])
    ap.add_argument("--mags", nargs="+", type=float, default=[0.70, 0.80, 0.90])
    ap.add_argument("--dur", type=float, default=0.30, help="持续推力时长（秒），同 viewer 默认")
    ap.add_argument("--seeds", type=int, default=60); ap.add_argument("--seed0", type=int, default=6000)
    ap.add_argument("--warmup", type=int, default=60); ap.add_argument("--horizon", type=int, default=250)
    ap.add_argument("--reset-noise", type=float, default=0.03)
    args = ap.parse_args()

    mod = load_mod((ROOT / args.dir).resolve())
    policy = build_policy(args.ckpt)
    seeds = list(range(args.seed0, args.seed0 + args.seeds))
    print(f"持续推力语义: 总冲量摊到 {args.dur:.2f}s (每步 {args.dur / 0.02:.0f} 步)  "
          f"seeds={args.seeds} reset_noise={args.reset_noise}")
    print(f"{'angle':>6} {'mag':>6} {'存活%':>7} {'峰值倾角':>9}")
    for a in args.angles:
        for m in args.mags:
            res = [run_one(make_env(mod, s, args.reset_noise), policy, a, m, args.dur, s,
                           args.warmup, args.horizon) for s in seeds]
            alive = 100 * np.mean([r[0] for r in res])
            peak = np.mean([r[1] for r in res])
            print(f"{a:6.0f} {m:6.2f} {alive:7.0f} {peak:9.1f}", flush=True)


if __name__ == "__main__":
    main()
