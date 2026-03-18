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
│   └── Examples/cartpole/
│       └── fault_oracle.py        CartPole 故障判定（functional / reward）
├── DI-engine/                     DreamerV3 训练后端
├── experiments/
│   └── dreamerv3_rq1/
│       ├── train_dreamerv3.py      训练 DreamerV3
│       ├── run_rq1.py             Path A：STARLA 随机变异
│       └── checkpoints/           训练产物
├── experiments_sam/
│   ├── run_four_experiments.py    四实验可行性套件（推荐入口）
│   ├── run_sam_experiment.py      Path B：单次 SAM 实验
│   ├── adapters/sam_mutation.py   SAMGuidedMutator（梯度引导变异）
│   └── results/                   feasibility_report.json、可视化
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
# 需先训练 DreamerV3
python experiments/dreamerv3_rq1/train_dreamerv3.py

# 运行四实验套件（单种子，约 30–45 分钟）
cd /workspaces/RL_Testing && python experiments_sam/run_four_experiments.py

# 多种子（默认 42，可改为 "42,123"）
python experiments_sam/run_four_experiments.py --seeds 42,123

# 自定义参数
python experiments_sam/run_four_experiments.py --population-size 20 --num-generations 20 --time-budget-seconds 300
```

### 输出

- `experiments_sam/results/feasibility_report.json`：聚合报告
- `experiments_sam/results/report_viz.html`：可视化（需本地 HTTP 服务打开）
- `experiments_sam/results/feasibility_report_seed_*.json`：单 seed 详情

---

## 最新实验结果（CartPole-v1，seed=42）

### 配置

- 功能性故障：strict OR 逻辑（position > 1.5 或 angle > 8°）
- Reward 故障：total_reward < 30
- MOSA 阈值：(100, 0.8, 0.8)，不提前终止

### 四组实验对比

| 实验 | ρ | search_reward_fault_rate | search_func_fault_rate | 耗时(s) |
|------|---|--------------------------|------------------------|---------|
| Exp1 STARLA only | — | 0.638 | 1.0 | 252 |
| Exp2 SAM best ρ | 0.05 | 0.676 | 1.0 | 247 |
| Exp3 SAM ½ρ | 0.025 | 0.638 | 1.0 | 318 |
| Exp4 SAM 2×ρ | 0.1 | **0.612** | 1.0 | 363 |

### 主要结论

1. **search_reward_fault_rate 有区分度**：Exp4（SAM 2×ρ）最低 0.612，说明较大 ρ 的 SAM 变异产生的 reward 故障最少；Exp2 略高于 baseline。
2. **search_func_fault_rate 全为 1.0**：MOSA 搜索偏向失败型 episode，CartPole 失败时 position 或 angle 必超阈值，当前 strict 定义下无法区分。
3. **SAM 稳定**：`sam_fallback_count = 0`，梯度计算均成功。

### 指标说明

- **search_reward_fault_rate**：搜索过程中 episode 的 reward 故障比例（reward < 30 步）
- **search_func_fault_rate**：搜索过程中 episode 的功能性故障比例（越界）
- **starla_archive_size**：MOSA 存档大小（通常 = num_objectives = 3 或受阈值限制）

---

## 其他入口

```bash
# Path A：STARLA 随机变异
python experiments/dreamerv3_rq1/run_rq1.py

# Path B：单次 SAM 实验
python experiments_sam/run_sam_experiment.py
```

---

## 依赖

```bash
pip install -e STARLA/
pip install easydict transformers tensorboardX
```

---

## 原始 README

完整框架说明、协议定义、DreamerV3 适配细节等见 **README_original.md**。
