#!/usr/bin/env python3
"""两个 checkpoint 的权重插值（用于把"前向专家"和"后向专家"结合成一个）。

用法:
  python blend_ckpt.py --a <A.pt> --b <B.pt> --out <目录> --alphas 0.2 0.3 ... 0.8
  # alpha=0 → 纯 A；alpha=1 → 纯 B

注意（项目历史教训）：v22best↔v25 的插值曾两次失败，没有任何 alpha 优于两端。
所以本工具只负责生成，**是否有效必须用 60-seed 评测判定**，不要凭 alpha 直觉交付。
"""
from __future__ import annotations
import argparse
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="A 权重（alpha=0 端）")
    ap.add_argument("--b", required=True, help="B 权重（alpha=1 端）")
    ap.add_argument("--out", required=True, help="输出目录")
    ap.add_argument("--alphas", nargs="+", type=float,
                    default=[0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    ap.add_argument("--prefix", default="blend")
    args = ap.parse_args()

    da = torch.load(args.a, map_location="cpu", weights_only=False)
    db = torch.load(args.b, map_location="cpu", weights_only=False)
    sa, sb = da["model_state_dict"], db["model_state_dict"]
    missing = set(sa) ^ set(sb)
    assert not missing, f"state_dict 键不一致: {sorted(missing)[:5]}"

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    for al in args.alphas:
        merged = {k: (1.0 - al) * sa[k].float() + al * sb[k].float() for k in sa}
        ck = {**da, "model_state_dict": merged}
        p = out / f"{args.prefix}_a{int(round(al * 100)):02d}.pt"
        torch.save(ck, p)
        print("wrote", p)


if __name__ == "__main__":
    main()
