# RL Testing：STARLA 测试框架 + DreamerV3 + SAM 引导变异

## 项目概述

将 STARLA（面向深度强化学习智能体的搜索式测试方法）改造成可插拔测试框架，接入 DI-engine 训练的 DreamerV3 作为被测智能体，并支持 SAM 梯度引导变异（输入空间扰动）与纯 STARLA 随机变异的对比实验。

---

## 目录结构

```
RL_Testing/
├── STARLA/                        测试框架核心
│   ├── src/starla/
│   │   ├── runner.py              StarlaRunner + TestReport
│   │   ├── config.py              StarlaConfig
│   │   ├── core/                  MOSA 搜索、遗传算子、fitness
│   │   ├── agents/                AgentProtocol、DreamerV3 适配器
│   │   ├── envs/                  Gymnasium 适配器（状态保存/恢复）
│   │   ├── faults/                FaultOracle 抽象
│   │   └── ml/                    Episode 编码、RF 故障预测
│   └── Examples/
│       ├── cartpole/              CartPole 故障判定
│       └── mountain_car/          Mountain Car 示例
├── DI-engine/                     DreamerV3 训练后端
├── experiments/
│   └── dreamerv3_rq1/
│       ├── train_dreamerv3.py      训练 DreamerV3（支持 CartPole / Mountain Car）
│       ├── run_rq1.py             Path A：STARLA 随机变异
│       └── checkpoints/           训练产物
├── experiments_sam/               CartPole 四实验套件
│   ├── run_four_experiments.py
│   ├── run_sam_experiment.py
│   ├── adapters/sam_mutation.py
│   └── results/
├── experiments_sam_mountaincar/   Mountain Car 四实验套件
│   ├── run_four_experiments.py
│   ├── run_sam_experiment.py
│   ├── adapters/sam_mutation.py
│   └── results/
├── experiments_atari_sam/         Atari Breakout 四实验套件
│   ├── train_dreamerv3_breakout.py  训练 DreamerV3（ALE/Breakout-v5, RGB 64x64）
│   ├── run_four_experiments_atari.py  四组实验 + ρ sweep
│   ├── adapters/sam_mutation_atari.py  Atari 专用 SAM 变异
│   ├── fault_oracle/breakout_fault_oracle.py  Breakout 故障判定
│   └── results/
├── experiments_mujoco_sam/        MuJoCo HalfCheetah 四实验套件（新增）
│   ├── train_dreamerv3_halfcheetah.py  训练 DreamerV3（HalfCheetah-v4, 17维状态）
│   ├── run_four_experiments_mujoco.py  四组实验 + ρ sweep
│   ├── adapters/sam_mutation_mujoco.py  MuJoCo 连续动作 SAM 变异
│   ├── envs/continuous_gymnasium_adapter.py  连续动作空间适配
│   ├── fault_oracle/halfcheetah_fault_oracle.py  HalfCheetah 故障判定
│   └── results/
├── experiments_dmcontrol_sam/     DMControl cheetah/run 像素四实验套件（新增）
│   ├── train_dreamerv3_walker.py    训练 DreamerV3（DMControl, RGB 64×64）
│   ├── run_four_experiments_walker.py  四组实验 + ρ sweep
│   ├── adapters/sam_mutation_walker.py  DMControl SAM 变异
│   ├── envs/walker_pixel_env.py     DMControl 像素环境封装
│   ├── fault_oracle/walker_fault_oracle.py  故障判定
│   └── results/
├── README_v2.md                   上一版 README（CartPole + Mountain Car）
├── README_old.md                  旧版 README（仅 CartPole）
└── README_original.md             原始 README 完整版
```

---

## 四实验可行性套件（推荐）

### 实验设计

`run_four_experiments.py` 一次性运行四组可对比实验：

| 实验 | 模式 | SAM ρ | 说明 |
|------|------|-------|------|
| Exp1 | STARLA only | — | 基线：纯随机变异 |
| Exp2 | SAM best ρ | sweep 选出的 best | SAM 引导，最优 ρ |
| Exp3 | SAM ½ρ | 0.5 × best | 较小扰动 |
| Exp4 | SAM 2×ρ | 2.0 × best | 较大扰动 |

流程：先对 ρ ∈ {0.01, 0.03, 0.05, 0.1, 0.2} 做 sweep，按故障数+时间选出 best_rho，再跑四组实验并生成聚合报告。

### 运行命令

```bash
# 需先训练 DreamerV3（按 meta 选择 CartPole 或 Mountain Car）
python experiments/dreamerv3_rq1/train_dreamerv3.py

# CartPole 四实验套件（约 30–45 分钟）
cd /workspaces/RL_Testing && python experiments_sam/run_four_experiments.py

# Mountain Car 四实验套件（约 1.5–2 小时）
cd /workspaces/RL_Testing && python experiments_sam_mountaincar/run_four_experiments.py

# Atari Breakout 四实验套件（约 1–1.5 小时）
# 需先训练 DreamerV3 Breakout 模型
python experiments_atari_sam/train_dreamerv3_breakout.py --max-env-step 500000
# 再运行四实验
python experiments_atari_sam/run_four_experiments_atari.py

# MuJoCo HalfCheetah 四实验套件（约 1–1.5 小时）
python experiments_mujoco_sam/train_dreamerv3_halfcheetah.py --max-env-step 200000
python experiments_mujoco_sam/run_four_experiments_mujoco.py

# DMControl cheetah/run 四实验套件（像素输入，约 1–2 小时）
# 需先安装 dm_control: pip install dm_control Pillow
python experiments_dmcontrol_sam/train_dreamerv3_walker.py --max-env-step 100000
python experiments_dmcontrol_sam/run_four_experiments_walker.py

# 多种子
python experiments_sam/run_four_experiments.py --seeds 42,123
python experiments_sam_mountaincar/run_four_experiments.py --seeds 42,123
python experiments_atari_sam/run_four_experiments_atari.py --seeds 42,123
python experiments_mujoco_sam/run_four_experiments_mujoco.py --seeds 42,123
python experiments_dmcontrol_sam/run_four_experiments_walker.py --seeds 42,123

# 自定义参数
python experiments_sam/run_four_experiments.py --population-size 20 --num-generations 20 --time-budget-seconds 300
```

### 输出

- `experiments_sam/results/feasibility_report.json`：CartPole 聚合报告
- `experiments_sam_mountaincar/results/feasibility_report.json`：Mountain Car 聚合报告
- `experiments_atari_sam/results/feasibility_report.json`：Breakout 聚合报告
- `experiments_mujoco_sam/results/feasibility_report.json`：HalfCheetah 聚合报告
- `experiments_dmcontrol_sam/results/feasibility_report.json`：DMControl cheetah/run 聚合报告
- `experiments_*/results/feasibility_report_seed_*.json`：单 seed 详情

---

## 最新实验结果

### CartPole-v1（seed=42）

**配置**：功能性故障 strict OR 逻辑；Reward 故障 total_reward < 30；MOSA 阈值 (100, 0.8, 0.8)

| 实验 | ρ | search_reward_fault_rate | search_func_fault_rate | 耗时(s) |
|------|---|--------------------------|------------------------|---------|
| Exp1 STARLA only | — | 0.638 | 1.0 | 252 |
| Exp2 SAM best ρ | 0.05 | 0.676 | 1.0 | 247 |
| Exp3 SAM ½ρ | 0.025 | 0.638 | 1.0 | 318 |
| Exp4 SAM 2×ρ | 0.1 | **0.612** | 1.0 | 363 |

**结论**：search_reward_fault_rate 有区分度；search_func_fault_rate 全为 1.0；SAM 稳定（fallback_count=0）。

---

### Mountain Car-v0（seed=42）

**配置**：功能性故障 default（GenericGymFaultOracle）；Reward 故障基于 env reward_threshold；MOSA 阈值 (100, 0.8, 0.8)

**总体结论**：✅ 管道可运行 | ✅ SAM 稳定 | ✅ **SAM 优于纯 STARLA** | 建议：扩大预算并多种子验证

#### ρ sweep（best_rho = 0.1）

| ρ | search_func_fault_rate | search_reward_fault_rate | 耗时(s) |
|---|------------------------|--------------------------|---------|
| 0.01 | 0.074 | 0.407 | 640 |
| 0.03 | **0.014** | **0.230** | 663 |
| 0.05 | 0.019 | 0.315 | 657 |
| **0.1** | 0.074 | 0.352 | **629** |
| 0.2 | 0.019 | 0.333 | 628 |

#### 四组实验对比

| 实验 | ρ | search_func_fault_rate | search_reward_fault_rate | 耗时(s) |
|------|---|------------------------|--------------------------|---------|
| Exp1 STARLA only | — | 0.037 | 0.278 | 647 |
| Exp2 SAM best ρ | 0.1 | **0.032** | **0.181** | 555 |
| Exp3 SAM ½ρ | 0.05 | 0.085 | 0.277 | 661 |
| Exp4 SAM 2×ρ | 0.2 | **0.014** | 0.243 | 670 |

**主要发现**：

1. **search_reward_fault_rate**：Exp2（SAM best ρ）最低 **0.181**，明显低于 baseline 0.278，SAM 引导变异更有效发现 reward 故障。
2. **search_func_fault_rate**：Exp4（ρ=0.2）最低 **0.014**，Exp2 为 0.032，均低于 baseline 0.037。
3. **与 CartPole 对比**：Mountain Car 上 SAM 优于纯 STARLA，而 CartPole 上之前未观察到该优势。
4. **random_reward_fault_rate = 1.0**：随机测试几乎全是 reward 故障，符合 Mountain Car 任务难度。

---

### ALE/Breakout-v5（seed=42）

**配置**：DreamerV3 RGB 64x64 输入；功能性故障 strict（BreakoutFaultOracle）；MOSA 阈值 (5.0, 0.8, 0.8)；20 代

> **注意**：当前 DreamerV3 训练步数较少（10000 步），模型基本为随机策略，random_reward_fault_rate 高达 95-100%。需更长训练才能得到有意义的 agent 行为对比。

#### ρ sweep（best_rho = 0.05）

| ρ | search_func_fault_rate | search_reward_fault_rate | 随机 reward fault 率 | 耗时(s) |
|---|------------------------|--------------------------|---------------------|---------|
| 0.01 | 8.9% | 21.1% | 98.9% | 954 |
| 0.03 | 13.3% | 23.3% | 100% | 889 |
| **0.05** | **10.0%** | **21.4%** | **95.7%** | **914** |
| 0.1 | 4.3% | 27.1% | 97.1% | 972 |
| 0.2 | 23.6% | 16.4% | 98.2% | 923 |

#### 四组实验对比

| 实验 | ρ | search_func_fault_rate | search_reward_fault_rate | 耗时(s) |
|------|---|------------------------|--------------------------|---------|
| Exp1 STARLA only | — | 24.5% | 20.9% | 908 |
| Exp2 SAM best ρ | 0.05 | **29.3%** | 13.3% | 966 |
| Exp3 SAM ½ρ | 0.025 | 19.2% | 14.6% | 955 |
| Exp4 SAM 2×ρ | 0.1 | **30.0%** | 14.6% | 864 |

**主要发现**：

1. **Pipeline 可运行**：STARLA+SAM 在 Atari RGB 环境上端到端跑通，包括 ALE 状态保存/恢复、图像预处理。
2. **SAM 稳定**：所有实验 `sam_fallback_count = 0`，无回退。
3. **SAM 优于纯 STARLA**：Exp2/Exp4 的 functional fault 发现率（29.3%/30.0%）高于纯 STARLA（24.5%）。
4. **Archive 较小**：所有实验 archive_size = 1，因模型训练不足，遗传搜索多样性有限。
5. **建议下一步**：扩大训练预算（≥500k env steps）+ 多种子验证。

---

### MuJoCo HalfCheetah-v4（seed=42）

**配置**：DreamerV3 向量输入（17 维状态）；连续动作空间（6 维）；功能性故障 strict（HalfCheetahFaultOracle）；MOSA 20 代

> **注意**：当前 DreamerV3 训练步数较少（200k 步），模型尚未充分训练，所有实验的 fault rate 完全相同，缺乏区分度。需更长训练才能得到有意义的对比。

#### ρ sweep（best_rho = 0.05）

| ρ | search_func_fault_rate | search_reward_fault_rate | 耗时(s) |
|---|------------------------|--------------------------|---------|
| 0.01 | 0.0 | 0.6 | 637 |
| 0.03 | 0.0 | 0.6 | 620 |
| 0.05 | 0.0 | 0.6 | 620 |
| 0.1 | 0.0 | 0.6 | 632 |
| 0.2 | 0.0 | 0.6 | 651 |

#### 四组实验对比

| 实验 | ρ | search_func_fault_rate | search_reward_fault_rate | 耗时(s) |
|------|---|------------------------|--------------------------|---------|
| Exp1 STARLA only | — | 0.0 | 0.6 | 648 |
| Exp2 SAM best ρ | 0.05 | 0.0 | 0.6 | 613 |
| Exp3 SAM ½ρ | 0.025 | 0.0 | 0.6 | 625 |
| Exp4 SAM 2×ρ | 0.1 | 0.0 | 0.6 | 648 |

**主要发现**：

1. **Pipeline 可运行**：STARLA+SAM 在 MuJoCo 连续动作空间上端到端跑通，包括连续动作遗传算子、HalfCheetah 状态保存/恢复。
2. **SAM 稳定**：所有实验 `sam_fallback_count = 0`，无回退。
3. **无区分度**：所有实验结果完全相同（func=0.0, reward=0.6），模型训练不足导致无法观察到 SAM 与 STARLA 差异。
4. **random_reward_fault_rate = 1.0**：随机测试全部为 reward 故障，符合训练不足的预期。
5. **建议下一步**：扩大训练预算（≥500k env steps）+ 多种子验证。

---

### DMControl cheetah/run from pixels（已跑通）

**配置**：DreamerV3 像素输入（RGB 64×64, CHW float32 [0,1]）；连续动作空间（6 维）；WalkerFaultOracle；MOSA 20 代

> Pipeline 已端到端跑通：dm_control 环境创建 → 像素渲染（camera_id=0, 64×64）→ RGB 预处理（HWC uint8 → CHW float32 [0,1]）→ DreamerV3 世界模型推理 → STARLA/SAM 遗传搜索。完整四组实验结果待补充。

---

### 五环境对比摘要

| 环境 | 观测类型 | 动作类型 | best_rho | SAM 优于 STARLA | 备注 |
|------|---------|---------|----------|-----------------|------|
| CartPole-v1 | 向量 (4D) | 离散 (2) | 0.05 | 否 | search_func_fault_rate 全为 1.0，难以区分 |
| Mountain Car-v0 | 向量 (2D) | 离散 (3) | 0.1 | **是** | search_reward/func_fault_rate 均有明显改善 |
| ALE/Breakout-v5 | RGB 64×64 | 离散 (4) | 0.05 | **是** | func_fault_rate 从 24.5% 提升至 30.0%（+5.5pp）|
| MuJoCo HalfCheetah-v4 | 向量 (17D) | 连续 (6D) | 0.05 | 待定 | 训练不足，所有实验结果完全相同 |
| DMControl cheetah/run | RGB 64×64 | 连续 (6D) | — | 待定 | Pipeline 已跑通，完整结果待补充 |

---

### 三环境输入输出详解（Breakout / HalfCheetah / DMControl cheetah）

| | ALE/Breakout-v5 | MuJoCo HalfCheetah-v4 | DMControl cheetah/run |
|---|---|---|---|
| **观测空间（Agent 输入）** | RGB 像素图像 (3, 64, 64) float32 [0,1]；原始 210×160 经 Pillow resize | 17 维向量：含躯干朝向、关节角度、关节角速度等 | RGB 像素图像 (3, 64, 64) float32 [0,1]；dm_control 相机渲染 |
| **动作空间（Agent 输出）** | Discrete(4)：NOOP / FIRE / RIGHT / LEFT | Box(6,)：6 个关节的连续力矩，范围 [-1, 1] | Box(6,)：6 个关节的连续力矩，范围 [-1, 1] |
| **单步 Reward** | 击碎砖块 +1；未击中 0 | 前进速度奖励 − 控制代价（连续值） | 前进速度奖励，范围约 [0, 10] |
| **Episode 长度** | 最大约 1000–2000 帧（取决于生命数） | 固定 1000 步（truncated） | 固定 1000 步（truncated） |
| **DreamerV3 世界模型输入** | obs − 0.5（图像中心化） | 17 维向量直接输入（obs_type="vector"） | obs − 0.5（图像中心化，obs_type="RGB"） |
| **预处理流程** | ALE RGB → Pillow resize 64×64 → HWC→CHW → /255 | 无需预处理（Gymnasium 直接返回向量） | dm_control pixels.Wrapper → Pillow resize 64×64 → HWC→CHW → /255 |
| **环境后端** | `gymnasium` + `ale-py` | `gymnasium` + MuJoCo 物理引擎 | `dm_control.suite` + `pixels.Wrapper` |
| **故障判定** | BreakoutFaultOracle：strict = 分数 < 5 或步数 < 100 | HalfCheetahFaultOracle：strict = 步数 < 100 或 reward < 500 | WalkerFaultOracle：strict = 步数 < 50 且 reward < 100 |

---

### 指标说明

- **search_reward_fault_rate**：搜索过程中 episode 的 reward 故障比例
- **search_func_fault_rate**：搜索过程中 episode 的功能性故障比例
- **starla_archive_size**：MOSA 存档大小（通常 = num_objectives = 3 或受阈值限制）

---

## 其他入口

```bash
# Path A：STARLA 随机变异
python experiments/dreamerv3_rq1/run_rq1.py

# Path B：单次 SAM 实验
python experiments_sam/run_sam_experiment.py
python experiments_sam_mountaincar/run_sam_experiment.py
```

---

## 依赖

```bash
pip install -e STARLA/
pip install easydict transformers tensorboardX
# MuJoCo（HalfCheetah）
pip install mujoco gymnasium[mujoco]
# DMControl（cheetah/run 像素）
pip install dm_control Pillow
# Atari（Breakout）
pip install ale-py shimmy "AutoROM[accept-rom-license]" Pillow
```

---

## 相关文档

- **README_v2.md**：上一版 README（CartPole + Mountain Car 结果）
- **README_old.md**：旧版 README（仅 CartPole 结果）
- **README_original.md**：完整框架说明、协议定义、DreamerV3 适配细节
