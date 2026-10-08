#!/usr/bin/env python3
"""按推力方向量化"卸力迈步"特征：抬脚步数、沿推力方向位移、步态是否成形。

用途：区分"原地硬扛"（几乎不抬脚）与"迈步卸力"（抬脚+定向位移+随后站稳）。
用法: python step_profile.py --versions v4 v8 v18 v20 --mag 0.50
"""
from __future__ import annotations
import argparse, dataclasses, importlib.util, os, sys
from pathlib import Path
import numpy as np, torch

ROOT = Path(__file__).resolve().parent.parent

def load_env():
    """加载仓库根目录的规范环境模块。"""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("duck_env_canonical", ROOT / "duck_vec_env.py")
    mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod

def build_policy(ckpt):
    from tensordict import TensorDict
    from rsl_rl.modules import ActorCritic
    s = TensorDict({"policy": torch.zeros(1,51), "critic": torch.zeros(1,55)}, batch_size=[1])
    p = ActorCritic(s, {"policy":["policy"],"critic":["critic"]}, num_actions=14,
        actor_hidden_dims=[512,256,128], critic_hidden_dims=[512,256,128], activation="elu",
        actor_obs_normalization=True, critic_obs_normalization=True,
        init_noise_std=0.5, noise_std_type="scalar")
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    p.load_state_dict(ck["model_state_dict"]); p.eval(); return p

def profile(mod, pol, angles, mag, warmup, horizon, seeds):
    cfg = dataclasses.replace(mod.MicroCfg(task_mode="push_recovery"),
        noise_joint_pos=0., noise_joint_vel=0., noise_gyro=0., noise_gravity=0.,
        encoder_bias_range=0., action_delay_max=0,
        mass_scale_range=(1.,1.), friction_damping_range=(1.,1.), armature_range=(1.,1.))
    env = mod.DuckVecEnv(cfg, num_envs=1, seed=1000, device="cpu")
    env.total_steps = 10**8; env.push_timer[:] = 1e9
    dt = cfg.ctrl_dt
    res = {}
    for ang in angles:
        rows = []
        for sd in seeds:
            with torch.inference_mode():
                env.rng = np.random.default_rng(sd); obs = env.reset(); env.push_timer[:] = 1e9
                for _ in range(warmup): obs, *_ = env.step(pol.act_inference(obs))
                start = env.datas[0].qpos[:2].copy()
                env.datas[0].qvel[0] += mag*np.cos(np.radians(ang))
                env.datas[0].qvel[1] += mag*np.sin(np.radians(ang))
                ca, sa = np.cos(np.radians(ang)), np.sin(np.radians(ang))
                lf = 0; prevc = env.foot_contact_prev[0].copy()
                along_max = 0.; perp_max = 0.; fz_max = 0.; peak = 0.; alive = True
                restored = None
                for t in range(horizon):
                    obs, _, done, _ = env.step(pol.act_inference(obs))
                    c = env.foot_contact_prev[0]
                    if (~c).any() and prevc.all(): lf += 1
                    prevc = c.copy()
                    disp = env.datas[0].qpos[:2] - start
                    al = disp[0]*ca + disp[1]*sa; pe = -disp[0]*sa + disp[1]*ca
                    along_max = max(along_max, al); perp_max = max(perp_max, abs(pe))
                    fz_max = max(fz_max, max(float(env.datas[0].site_xpos[env._foot_site_ids[j],2]) for j in range(2)))
                    pg = env._proj_gravity[0]
                    peak = max(peak, float(np.degrees(np.hypot(pg[0], pg[1]))))
                    v = env._v_body[0]
                    if restored is None and t*dt > 0.4 and np.hypot(v[0],v[1]) < 0.05:
                        restored = t*dt
                    if bool(done.item()): alive = False; break
                rows.append(dict(lf=lf, along=along_max, perp=perp_max, fz=fz_max,
                                 peak=peak, alive=alive, rest=restored))
        agg = {k: np.mean([r[k] for r in rows]) * (100 if k == "alive" else 1)
               for k in rows[0] if k != "rest"}
        # 摔倒的回合不会记录恢复时刻，None 要剔除（全为 None 时输出 nan）
        rests = [r["rest"] for r in rows if r["rest"] is not None]
        agg["rest"] = float(np.mean(rests)) if rests else float("nan")
        res[ang] = agg
    return res

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--mag", type=float, default=0.50)
    ap.add_argument("--angles", nargs="+", type=float,
                    default=[0,45,90,135,180,225,270,315])
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--horizon", type=int, default=400)
    ap.add_argument("--seeds", type=int, default=6)
    args = ap.parse_args()
    seeds = list(range(3000, 3000+args.seeds))
    mod = load_env(); pol = build_policy(args.ckpt)
    res = profile(mod, pol, args.angles, args.mag, args.warmup, args.horizon, seeds)
    print(f"\n=== {os.path.basename(args.ckpt)}  push={args.mag} m/s ===")
    print(f"{'angle':>6} {'存活%':>7} {'抬脚数':>7} {'沿推力位移':>10} {'侧向位移':>9} "
          f"{'抬脚高度':>9} {'峰值倾角':>9} {'恢复耗时':>9}")
    for a in args.angles:
        r = res[a]
        print(f"{int(a):>6} {r['alive']:7.0f} {r['lf']:7.1f} {r['along']:10.3f} {r['perp']:9.3f} "
              f"{r['fz']:9.3f} {r['peak']:9.1f} {r['rest']:9.2f}")

if __name__ == "__main__":
    main()
