# microduck 原始实现调研：与我们这条线的差异（2026-09-18）

> 调研对象：`microtest_archive/_microduck_src/`（pollen-robotics microduck，mjlab 版）。
> 目的：为"卸力抗推"找可迁移的做法。结论：**它解决的是站起+行走，不是高幅抗推**；
> 但它的三条设计原则精确命中我们反复踩的坑。

---

## 发现 1（最重要）：恢复类奖励一律用「势函数塑形」，不用"状态给分"

**它的做法**（`tasks/mdp.py`）：

```python
def upright_progress(...):   # 权重 5.0
    """Potential-based upright shaping: Δcos(tilt) per step."""
    cos_tilt = 1.0 - 2.0 * (quat[:,1]**2 + quat[:,2]**2)
    delta = cos_tilt - env._upright_potential_prev       # 只结算"变化量"
    env._upright_potential_prev = cos_tilt.clone()
    return delta
```

```python
def height_progress(...):    # 权重 30.0，ceiling=0.115
    """Potential-based height shaping: Δ min(trunk z, ceiling) per step."""
    return pot - env._height_potential_prev
```

原文注释（`microduck_velstand_env_cfg.py`，逐字）：

> **"ANY positive reward for BEING in a fallen-ish state gets farmed from some
> comfortable pose. The orientation reward is therefore POTENTIAL-BASED
> (Δcos tilt): rising pays, falling costs, holding anything pays zero.
> Unfarmable, ungated, and also rewards catching a stumble while walking."**

> *"Potential-based shaping is policy-invariant (Ng et al.): it accelerates
> learning of recovery without creating new optima."*

它为此付出过代价（注释里的 run 1/2/4 教训）：**曾经用过 `+2·cos(tilt)` 这种
"越直立越给分"的状态奖励，结果被"坐姿/躺平/头顶三脚架"三种偷分姿态农场化**，
最后只能靠"头部撞击惩罚"歪打正着地挡住 —— 去掉那个惩罚后立刻复发。

**对照我们这条线**：v26 起我们加的全是**状态给分**：

```python
yield_step = react * (~both_feet_air) * (tilt_deg < 35°) * 单脚离地   # 抬脚就给
yield_walk = react * back_gate * clip(v_along / 0.6)                  # 顺着让位就给
catch_step = react * [(落地点在推力方向一侧) × 偏移量]                 # 垫步就给
```

**这些奖励在"身体已经开始倒、但还没到 35°"的区间里照样发满分。**
v35/v44 的失败轨迹正是这样：抬脚后倾角从 12° 一路单调涨到 34–38°（还在拿
yield_step 的分），然后摔。微观上就是"奖励在鼓励它一边倒一边抬脚"。

**这是我们后向打不上去的一个具体、可修的原因**（不是全部原因，见发现 4）。

---

## 发现 2：microduck 主动把推力从 ±0.5 降回 ±0.3，并说明了理由

`microduck_velocity_env_cfg.py` 原文：

> **"VELOCITY_PUSH_RANGE = (-0.3, 0.3)  # Was ±0.5 — an ADDITIVE kick larger
> than max walk speed (0.4) every 3-6 s trains a permanently nervous
> fall-recovery gait (2026-07 audit). ±0.3 keeps push robustness while
> letting a calmer gait be optimal."**

配套参数：
- `VELOCITY_PUSH_INTERVAL_S = (3.0, 6.0)` —— 推力间隔 **3–6 秒**（我们 v27+ 一路压到
  0.8–1.6s，v26 是 1.2–2.5s）
- `ENABLE_VELOCITY_PUSHES = True` 但**课程**是 `0 → ±0.08 → ±0.3`（按 step 分三段）
- 语义：`mdp.push_by_setting_velocity`（**瞬时设速**，与我们一致，**不是**持续推力）

**它的取舍**：为了让"平静步态"成为最优解，主动把推力压在行走速度以下。
**我们的处境相反**：评测压到 0.70/0.75（≈行走速度的 1.75–1.9 倍），是分布外外推
—— 这一点 CLAUDE.md 里早有记录。

---

## 发现 3：静止站姿用「三项相乘」的复合分 + 离散达标奖

```python
def standing_composite_score(...):
    return height_score * upright_score * pose_score   # 三项高斯相乘
def standing_success_bonus(...):
    return (height_ok & upright_ok & pose_ok).float()  # 全达标才给
```

注释说明动机：

> *"Because the factors **multiply**, a deficiency in any one term collapses the
> whole reward — the policy can't claim 80% of this by being perfect on 2-of-3.
> Use to break Nash-equilibrium compromises (e.g., a **'lean trunk at the right
> height' basin** that satisfies the additive rewards' partial sums)."*

**这精确对应我们 v27/v30/v31 的病**：站立目标拆成若干加性项后，策略找到
"歪着身子凑高度/凑姿态"的折中盆地（v27 静止倾角 11.5°、v27c 17.0°、v30 13.9°、
v31 18.2°）。microduck 用"相乘 + 达标奖"来消灭这类盆地。
我们最终靠 `pose_probe` 量出 v26_17100 是唯一"直"的（3.42°），但那是运气，不是设计。

---

## 发现 4：它没有解决高幅抗推 —— 我们面对的是它明确回避的问题

microduck 全部任务里 `VELOCITY_PUSH_RANGE` 最大值是 **±0.3**（行走类），
姿态类任务主动降到 ±0.15/±0.2。它没有任何一处把推力顶到行走速度以上并
"既要平静站姿又要抗住"。

**所以：不能指望照搬它的参数解决问题。** 能照搬的是**方法**：
势函数塑形（发现 1）、乘性复合分（发现 3）、以及"推力要配得起步态"的取舍意识（发现 2）。

---

## 发现 5：其它可直接借用的工程手法

| 手法 | 出处 | 作用 |
|---|---|---|
| `feet_air_time_upright` | `mdp.py:527` | 摔倒（tilt>40°）时把抬脚奖励清零 —— 治"躺在地上抖腿刷分" |
| `fallen_state_penalty` + 迟滞释放 | `velstand` | 倒下持续扣分，**直到真正站好才释放**（避免"蹲在门槛下白拿"） |
| `recovery_success` 一次性赏金 | `mdp.py:646` | 只在"完成恢复"那一帧给，迟滞防抖 |
| `joint_torque_rate_l2` | `mdp.py` | 罚**力矩变化率**（不是幅值/动作），平滑转移又不阻止恢复动作 |
| 课程按 `step` 而非 iter 分三段 | `push_curriculum` | 推力 `0 → ±0.08 → 最终`，避免开局被推崩 |

---

## 结论：下一步该改什么（按优先级）

1. **把"抬脚/让位/垫步"从状态奖励改成势函数**（发现 1）——
   `Δcos(tilt)` 式：正在被扶正 → 给分；正在继续倒 → 扣分；保持不动 → 零分。
   直接掐掉"一边倒一边拿卸力分"的偷分路径。
2. **站立评分改成乘性复合 + 达标赏金**（发现 3），替代现在的加性姿态项。
3. **推力与步态匹配**（发现 2）：若继续追 0.70+，必须接受步态更紧张；
   或按它的路线退到 0.3–0.4 换平静步态（但那与"抗推更强"的目标冲突，需用户决策）。
4. **补上 `feet_air_time_upright` / 倒下迟滞扣分**（发现 5）把"躺着刷分"堵死。


---

## 验证结果（v46 / v47，2026-09-18）

按发现 1 做了两版实验，**结论：势函数塑形没有解决"前后互斥"，但对后向专家无害**。

### v46：后向专家 + 势函数塑形
从 `_v44_1800` 出发（保持 v44 的 70% 后向配比）续训 1200 iter。
结果：后向保住了（多个检查点持续语义 0.70/0.80 双 100%），但前向仍是 0
—— 因为配比没变，前向本来就没数据，这个实验不构成对"塑形"的检验。

### v47：与 v45 的**干净对照**（唯一差异 = 势函数塑形）

| | v45（无塑形） | v47（有塑形） |
|---|---|---|
| 采样配比 | fwd 0.55 / bwd 0.25 | **完全相同** |
| 后向折减 | 0.90 | **完全相同** |
| 课程 | 0.56→0.62→0.66 | 0.56→0.64→0.70 |
| 起点 | v44_1800 | **完全相同** |
| **持续语义 后向 0.80** | 全程 0 | 少数点 30/100/5（不稳定） |
| **持续语义 后向 0.70** | 末段 35 | 末段 5 |
| **前向 0.70** | 末段 25 | 末段 0 |

**结论**：塑形**没有**把前向救回来，也没有稳定保住后向的高幅能力。
加上"在窗口内倾角正在回落才给卸力分"的门控后，策略反而更难学到动作
（大量检查点全项为 0）。说明发现 1 的机理诊断**可能仍然正确**（"边倒边拿分"
确实是偷分路径），但**单独改这一项不足以突破前后互斥** —— 因为互斥的根源
是物理动作模式冲突（前向要绷住、后向要松开），不是奖励项的形式。

### 仍然成立的部分

- 发现 1（势函数）**对"后向专家"无害且合理**：v46 在保持 70% 后向配比时，
  后向 0.70/0.80 的高存活率与 v44 相当。若将来做后向专家，建议保留这个塑形。
- 发现 3（乘性复合分）**尚未在本项目验证** —— 值得用于治"歪着凑高度"的站姿，
  这是独立于抗推的一个问题（v27/v30/v31 的 11–18° 静止倾角）。
- 发现 2（推力与步态匹配）**已被本项目反复验证**：v16/v27b/v29/v30/v36/v42
  六次"加大推力/加密推力"全部失败，与 microduck"±0.5 会训出神经质步态"
  的结论完全一致。

### 建议的下一步

放弃在"单模型同时兼顾前后高幅"上继续投入（已 18 版实验证否），转向：
1. **交付两个模型按场景切换**（`_v44_1800` 后向 / `_v35_17700` 前向），
   或用 `v38_17725` 作单一折中；
2. 若必须单模型，用发现 3 的乘性复合分 + 发现 5 的堵漏项重做**站姿**，
   但把抗推目标降到 microduck 的建议区间（≤0.4），换平静步态。

---

# 补充调研（2026-09-18 第二轮）：microduck 如何处理"别乱动"与"被推恢复"

## 发现 6（重要）：**它不是靠惩罚项防踏步，而是靠"任务配比"**

`microduck_velocity_env_cfg.py:330` 逐字：

> *"air_time window [0.125, 0.300] s. NOTE: **standing still at zero command is
> taught by the standing_envs curriculum (→25% standing envs by ~iter 2000),
> not by an explicit stillness/no-stepping term.**"*

它的实现（`mdp.py:3295 standing_envs_curriculum`）：

```python
{"step": 0,        "rel_standing_envs": 0.02},
{"step": 500*24,   "rel_standing_envs": 0.05},
{"step": 750*24,   "rel_standing_envs": 0.10},
{"step": 1000*24,  "rel_standing_envs": 0.15},
{"step": 1500*24,  "rel_standing_envs": 0.20},
{"step": 2000*24,  "rel_standing_envs": 0.25},   # 最终 25% 环境专做纯站立
```

注释还写明：`rel_standing_envs = 0.02  # small but non-zero from the start`

**对照我们**：`zero_command_prob` 从 0.50 **一路递减到 0.10**，而这 0.10 还是
"指令为零"——但 push_recovery 任务里**所有指令本来就是 0**，这个字段实际不起作用。
真正在做防踏步的是惩罚项（`w_idle_lift`、`w_idle_legact`），而这正是**没效果且有害**
的那条路：renew2 已实测证明"惩罚到不踏步 ⟹ 后向归零"。

**结论**：正确做法是**留出一部分环境专门做纯站立任务**（不推、只站着收分），
让"安静站立"成为一条被独立优化的技能，而不是让它在抗推任务里被惩罚出来。

## 发现 7：它交接抬脚动作/抬脚高度的具体数值

| 项 | microduck | 我们（renew2） | 差异 |
|---|---|---|---|
| `air_time` 权重 | **3.0** | 3.0 | 同 |
| air_time 窗口 | **[0.125, 0.300] s** | [0.10, 0.25] | 我们的窗口偏短、上限低 |
| `foot_clearance.y` | — | — | — |
| `foot_clearance.target_height` | 0.02（注释 *"penalize dragging"*） | — | 同量级 |
| `foot_swing_height.target_height` | 0.02（注释 *"**force foot lifting**"*） | `foot_swing_target`=0.02 | 同 |
| `body_ang_vel` | **-0.05** | `w_ang_vel_xy`=-0.1 | 我们严一倍 |
| `angular_momentum` | **-0.02** | 无 | **我们缺这一项** |
| 指令范围 | **固定**（lin±0.4 / ang±1.0） | 课程渐宽 | 它刻意不扩宽（"扩宽超出了机器人能力"） |

## 发现 8：它的速度任务本来就没有"被推"这个子任务

microduck 的 push 只作为**扰动**（3–6s 一次、±0.3）叠在行走任务上，
没有"被推后恢复"的独立奖励层（那是 velstand/standup 任务的职责，且是
从**躺倒**恢复、不是抗住站姿）。所以它的 25% 站立环境 + ±0.3 推力，
与我们要的"0.70 高幅抗推"本来就不是同一个问题。

**对下一步的意义**：可以照搬它的"站立环境配额"机制（发现 6），
但推力幅值必须我们自己定（不能照搬 ±0.3）。
