#!/usr/bin/env python3
"""生成 README 里的抗推对比图（数据全部来自 60-seed 评测）。

数据来源：
  - docs/v35_chart_data_s60.txt        v35_17700 / v35_18175 的 60-seed 表
  - docs/V36_EXPERIMENT.md             v26_17100 基线、step_profile 机制指标

用法: python charts/make_charts.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

plt.rcParams["font.sans-serif"] = ["PingFang HK", "Hiragino Sans GB", "Heiti TC", "Arial Unicode MS"]
plt.rcParams["axes.unicode_minus"] = False

OUT = Path(__file__).resolve().parent / "push_recovery_comparison.png"

# ---- 数据（60 seeds, seed0=6000, reset_noise=0.03, 瞬时冲量语义）----
V35, V26 = "v35_17700", "v26_17100"
SURVIVAL = {
    "0 deg @ 0.70 m/s":   {V35: 98, V26: 90},
    "0 deg @ 0.75 m/s":   {V35: 82, V26: 58},
    "180 deg @ 0.55 m/s": {V35: 88, V26: 50},
    "180 deg @ 0.60 m/s": {V35: 22, V26: 45},
}
# step_profile, 10 seeds
MECHANISM = {
    "foot lifts (count)":    {V35: 2.9, V26: 1.8},
    "peak tilt (deg)":       {V35: 10.6, V26: 15.3},
    "drift (cm)":            {V35: 4.2, V26: 5.3},
}

FORWARD = (0, 1)
V35_COLOR = "#1f4e79"
V26_COLOR = "#9dc3e6"
ACCENT = "#e8873a"


def _bar_panel(ax, metrics, title, xlabel, ylabel):
    labels = list(metrics)
    ys = range(len(labels))
    h = 0.36
    for i, label in enumerate(labels):
        v35 = metrics[label][V35]
        v26 = metrics[label][V26]
        ax.barh(i + h / 2, v35, height=h, color=V35_COLOR, zorder=3)
        ax.barh(i - h / 2, v26, height=h, color=V26_COLOR, zorder=3)
        ax.text(v35 + 1.2, i + h / 2, f"{v35:g}", va="center", fontsize=9, color=V35_COLOR)
        ax.text(v26 + 1.2, i - h / 2, f"{v26:g}", va="center", fontsize=9, color="#5a7fa0")
    ax.set_yticks(list(ys))
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left", fontsize=12, fontweight="bold")
    ax.grid(axis="x", color="#dddddd", zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    ax = axes[0]
    _bar_panel(ax, SURVIVAL, "Survival rate by push scenario",
               "survival rate (%)", "push scenario")
    # 阈值参考线：前向 0.70 需 >= 95%，后向 0.55 需 >= 85%
    ax.axvline(95, color=ACCENT, ls="--", lw=1.2, zorder=4)
    ax.axvline(85, color=ACCENT, ls=":", lw=1.2, zorder=4)
    ax.text(96, -0.42, "95% forward gate", color=ACCENT, fontsize=8, va="center")
    ax.text(86, 3.55, "85% backward gate", color=ACCENT, fontsize=8, va="center")

    ax = axes[1]
    _bar_panel(ax, MECHANISM, "Yield mechanism (180 deg @ 0.55 m/s, 10 seeds)",
               "value", "")
    ax.set_xlim(0, 18)

    legend = [
        Patch(facecolor=V35_COLOR, label="v35_17700 (recommended)"),
        Patch(facecolor=V26_COLOR, label="v26_17100 (previous best)"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("OpenDuckMini push recovery: v35_17700 vs v26_17100",
                 fontsize=14, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
