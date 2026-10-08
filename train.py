#!/usr/bin/env python3
"""microtest 训练入口 — rsl_rl OnPolicyRunner + DuckVecEnv。

用法:
    python train.py                          # 正式训练: 128 环境 × 3000 迭代
    python train.py --smoke                  # 冒烟: 16 环境 × 5 迭代（训练前必跑）
    python train.py --resume logs/<run>/model_1000.pt
    mjpython play.py <checkpoint>            # 可视化（后续提供）

超参取自 microduck_rl 的配方（见 TRAINING_PLAN.md 第 7 节）。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from rsl_rl.runners import OnPolicyRunner

from duck_vec_env import CURRICULUM, DuckVecEnv, MicroCfg, NUM_STEPS_PER_ENV


class MicroRunner(OnPolicyRunner):
    """在标准日志之外，注入环境统计（回合回报/长度/吞吐）。"""

    def log(self, locs: dict, width: int = 80, pad: int = 35) -> None:
        if getattr(self, "writer", None) is not None:
            stats = self.env.pop_episode_stats()
            if stats:
                self.writer.add_scalar("Episodes/return_mean", stats["return_mean"], locs["it"])
                self.writer.add_scalar("Episodes/length_mean", stats["length_mean"], locs["it"])
                self.writer.add_scalar("Episodes/count", stats["count"], locs["it"])
            stage = self.env.stage()
            self.writer.add_scalar("Curriculum/stage_idx", CURRICULUM.index(stage), locs["it"])
        super().log(locs, width, pad)


def make_train_cfg(seed: int, learning_rate: float, save_interval: int) -> dict:
    return {
        "seed": seed,
        "num_steps_per_env": NUM_STEPS_PER_ENV,
        "save_interval": save_interval,
        "logger": "tensorboard",
        "experiment_name": "micro_velocity",
        "run_name": "micro_velocity",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {
            "class_name": "ActorCritic",
            "actor_hidden_dims": [512, 256, 128],
            "critic_hidden_dims": [512, 256, 128],
            "activation": "elu",
            "actor_obs_normalization": True,
            "critic_obs_normalization": True,
            "init_noise_std": 0.5,
            "noise_std_type": "scalar",
        },
        "algorithm": {
            "class_name": "PPO",
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "clip_param": 0.2,
            "gamma": 0.99,
            "lam": 0.95,
            "value_loss_coef": 1.0,
            "entropy_coef": 0.01,
            "learning_rate": learning_rate,
            "max_grad_norm": 1.0,
            "use_clipped_value_loss": True,
            "schedule": "adaptive",
            "desired_kl": 0.01,
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--max-iters", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--reset-course", action="store_true",
                        help="续训时从低推力课程重新开始，而不是沿用 checkpoint 迭代数")
    parser.add_argument("--task", choices=("stand", "push_recovery", "velocity", "jump"), default="velocity")
    parser.add_argument("--smoke", action="store_true", help="冒烟测试: 16 环境 × 5 迭代")
    parser.add_argument("--push-scale", type=float, default=1.0,
                        help="推力上限缩放系数（作用于课程最后一段的推力幅度）")
    parser.add_argument("--start-stage", type=int, default=None,
                        help="续训时从指定课程阶段起步（0 最低；不指定则沿用 checkpoint 迭代数）")
    parser.add_argument("--learning-rate", type=float, default=1.0e-3,
                        help="PPO 学习率；保守精调时建议 2e-4")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="检查点保存间隔（迭代数）")
    parser.add_argument("--resume-push-scale", action="store_true",
                        help="续训时沿用 checkpoint 记录的推力缩放（默认按 --push-scale）")
    args = parser.parse_args()

    if args.smoke:
        args.num_envs, args.max_iters = 16, 5

    torch.manual_seed(args.seed)
    env_cfg = MicroCfg(task_mode=args.task)
    env = DuckVecEnv(env_cfg, num_envs=args.num_envs, seed=args.seed, device="cpu")
    env.push_scale = args.push_scale

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp = "smoke" if args.smoke else f"micro_{args.task}"
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", f"{exp}_{stamp}")
    os.makedirs(log_dir, exist_ok=True)

    with open(os.path.join(log_dir, "env_config.json"), "w") as f:
        json.dump(env.as_dict(), f, indent=2, ensure_ascii=False)

    train_cfg = make_train_cfg(args.seed, args.learning_rate, args.save_interval)
    runner = MicroRunner(env, train_cfg, log_dir=log_dir, device="cpu")
    ckpt_push_scale = 1.0
    if args.resume:
        runner.load(args.resume)
        try:
            ckpt_push_scale = float(torch.load(args.resume, map_location="cpu",
                                               weights_only=False).get("push_scale", 1.0))
        except Exception:
            ckpt_push_scale = 1.0
        print(f"已从 {args.resume} 恢复")
        # 恢复后将环境步数对齐到当前迭代，避免课程（stage/action_rate）回退到第 0 阶段
        env.total_steps = (0 if args.reset_course else
                           runner.current_learning_iteration * NUM_STEPS_PER_ENV * args.num_envs)
        if args.resume_push_scale:
            env.push_scale = ckpt_push_scale
        if args.start_stage is not None:
            st = max(0, min(args.start_stage, len(CURRICULUM) - 1))
            env.total_steps = (CURRICULUM[st][0] * NUM_STEPS_PER_ENV * args.num_envs)
            print(f"课程起点: stage {st} (iter>={CURRICULUM[st][0]}, 推力上限 "
                  f"{CURRICULUM[st][6] * env.push_scale:.2f} m/s)")
    print(f"推力上限缩放: {env.push_scale:.2f}  (课程末段推力 = "
          f"{CURRICULUM[-1][6] * env.push_scale:.2f} m/s)")

    print(f"观测: policy={env._obs_policy.shape[1]}D critic={env._obs_critic.shape[1]}D")
    print(f"动作: {env.num_actions}D | 环境数: {env.num_envs} | 日志: {log_dir}")
    runner.learn(num_learning_iterations=args.max_iters, init_at_random_ep_len=False)

    final = os.path.join(log_dir, "model_final.pt")
    runner.save(final)
    # 记录本次运行的推力上限缩放，便于续训时保持一致
    try:
        ck = torch.load(final, map_location="cpu", weights_only=False)
        ck["push_scale"] = float(env.push_scale)
        torch.save(ck, final)
        for extra in ("model_%d.pt" % runner.current_learning_iteration,):
            pth = os.path.join(log_dir, extra)
            if os.path.exists(pth):
                c2 = torch.load(pth, map_location="cpu", weights_only=False)
                c2["push_scale"] = float(env.push_scale)
                torch.save(c2, pth)
    except Exception as exc:
        print(f"(未能写入 push_scale: {exc})")
    print(f"训练结束，最终模型: {final}")


if __name__ == "__main__":
    main()
