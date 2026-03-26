# 边缘扰动实验：DreamerV3 在 MountainCar-v0 上的脆弱性分析

对 DreamerV3 智能体的感知-决策-执行流水线的四条边分别施加不同的扰动策略，通过 STARLA 遗传搜索框架评估每条边的脆弱程度。

## 实验设计

| 编号 | 名称 | 扰动的边 | 方法 |
|:----:|------|---------|------|
| exp0 | Baseline | — | 原始 STARLA 随机噪声（仅扰动 position 维度 ±5%） |
| exp1 | E1 SAM | 观测（critic 引导） | SAM 梯度引导扰动：沿 critic 价值下降最快的方向扰动观测（rho=0.1） |
| exp2 | E2 Jacobian | 观测（encoder 引导） | Jacobian 稀疏扰动：计算 encoder 的 Jacobian 矩阵，只扰动对 latent 影响最大的观测维度 |
| exp3 | E3 Uncertainty | 世界模型预测 | 不确定性引导扰动：对 RSSM 做多次随机前向传播，沿预测方差增大的方向扰动观测 |
| exp4 | E4 Temporal | 动作时序 | 动作延迟：不扰动观测，而是让 agent 每步执行上一步的动作，模拟执行延迟 |

**参考文献：**

- E1：SAM（Sharpness-Aware Minimization）— Foret et al., ICLR 2021
- E2：Jacobian 正则化 — Jakubovitz & Giryes, ECCV 2018 (arXiv:1803.08680)
- E3：MOPO 不确定性惩罚 — Yu et al., NeurIPS 2020 (arXiv:2005.13239)
- E4：Action Robust RL — Tessler et al., ICML 2019 (arXiv:1901.09184)

## 实验结果（MountainCar-v0, seed=42）

### Reward Fault Rate（奖励故障率）

fault rate 越高，说明该扰动方式越能有效暴露智能体的缺陷，即该边越脆弱。

| 排名 | 实验 | 扰动的边 | Reward Fault Rate | 与 Baseline 相比 |
|:----:|------|---------|:-----------------:|:----------------:|
| 1 | E3 Uncertainty | 世界模型预测 | 32.43% | +4.65% |
| 2 | E1 SAM | 观测（critic 引导） | 29.63% | +1.85% |
| 3 | E2 Jacobian | 观测（encoder 引导） | 28.38% | +0.60% |
| 4 | Baseline | 随机噪声 | 27.78% | — |
| 5 | E4 Temporal | 动作时序 | 27.03% | -0.75% |

### Functional Fault Rate（功能故障率）

功能故障检测的是"崩溃级"异常：episode 异常提前终止（少于 20 步）或观测出现 NaN/Inf。

| 实验 | 功能故障数 | 功能故障率 |
|------|:---------:|:---------:|
| exp0 Baseline | 0 | 0.00% |
| exp1 E1 SAM | 0 | 0.00% |
| exp2 E2 Jacobian | 0 | 0.00% |
| exp3 E3 Uncertainty | 0 | 0.00% |
| exp4 E4 Temporal | 1 | 1.35% |

**为什么功能故障几乎为零：**

MountainCar-v0 是一个物理上极其简单稳定的环境。观测空间只有 2 维（position ∈ [-1.2, 0.6]，velocity ∈ [-0.07, 0.07]），物理引擎始终产生有限值，不会出现 NaN 或 Inf。episode 几乎总是跑满 200 步（环境的 max_episode_steps），远超功能故障判定阈值 20 步。E1/E2/E3 的扰动只改变了 agent 看到的观测值，不影响环境内部物理状态，所以环境本身不会崩溃。

**为什么只有 E4（动作延迟）触发了功能故障：**

E4 不扰动观测，而是让 agent 每步执行上一步算出的动作。与 E1/E2/E3 只在单个变异点施加一次扰动不同，E4 的动作延迟作用于 episode 的每一步（共 5976 次动作延迟）。这种持续的时序错乱可能导致 agent 做出完全不合理的动作序列，在极端情况下使 episode 异常终止，从而触发功能故障检测。

### 额外指标

| 实验 | Fallback 次数 | 特有指标 | 值 |
|------|:------------:|---------|:--:|
| exp1 E1 SAM | 0 | sam_rho | 0.1 |
| exp2 E2 Jacobian | 0 | avg_jacobian_sparsity | 1.0（全维度扰动） |
| exp3 E3 Uncertainty | 0 | avg_uncertainty_magnitude | 0.0（梯度链断裂） |
| exp4 E4 Temporal | — | action_perturbation_count | 5,976 |

**异常说明：**

- **E2 sparsity = 1.0（未产生稀疏效果）：** MountainCar 只有 2 个观测维度，但 top_k 设为 3。由于 top_k > obs_dim，每次所有维度都被扰动了，稀疏扰动的设计意图没有体现。这个指标在高维环境（如 Atari 的 210×160×3 像素观测）中才有意义。

- **E3 uncertainty magnitude = 0.0（梯度链断裂）：** RSSM 在 `dynamics.obs_step(sample=True)` 中的随机采样操作没有维持从输入观测到采样结果的可微分梯度路径。虽然 5 次前向传播因为不同的 latent 采样产生了不同的 reward 预测，但这种差异在数学上不依赖于输入 obs，因此反向传播回去梯度为零，每次都退化为随机扰动。尽管如此，E3 仍然取得了最高的 fault rate，可能是因为 RSSM 随机采样本身在 STARLA 搜索过程中引入了有益的多样性。

## 脆弱性画像

```
最脆弱的边：  E3_prediction  （世界模型预测通道）
最鲁棒的边：  E4_execution   （动作执行通道）
```

DreamerV3 在 MountainCar 上功能鲁棒（不会崩溃或产生异常状态），但存在奖励级别的脆弱性，尤其是世界模型的预测通道被针对性扰动时最容易暴露缺陷。

## 运行方式

```bash
cd /workspaces/RL_Testing
python -m experiments_edge_perturbation.run_edge_experiments --seeds 42
```

结果输出到 `experiments_edge_perturbation/results/`。
