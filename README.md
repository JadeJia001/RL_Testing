# RL Testing：把 STARLA 改造成测试框架并接入 DI-engine DreamerV3

## 我做了什么

这个项目的核心思路是：把 STARLA（一个面向深度强化学习智能体的搜索式测试方法）从"写死的 Notebook/脚本"形态，改造成一个可插拔的测试框架，再把 DI-engine 训练出来的 DreamerV3 作为被测智能体接入，跑端到端的测试实验。

---

## 目录结构

```
RL_Testing/
├── STARLA/                        改造后的测试框架核心
│   ├── src/starla/
│   │   ├── runner.py              框架总入口（StarlaRunner + TestReport）
│   │   ├── config.py              StarlaConfig（Pydantic 配置类）
│   │   ├── core/
│   │   │   ├── mosa.py            MOSA 多目标遗传搜索引擎
│   │   │   ├── genetic.py         遗传算子（crossover / mutate / re_execute）
│   │   │   ├── candidate.py       候选 episode 数据结构
│   │   │   └── fitness.py         三目标适应度函数
│   │   ├── agents/
│   │   │   ├── base.py            AgentProtocol 协议定义
│   │   │   └── sb3_adapter.py     SB3（DQN/PPO 等）适配器
│   │   ├── adapters/
│   │   │   └── dreamerv3/
│   │   │       └── agent.py       DreamerV3 适配器（DI-engine → STARLA）
│   │   ├── envs/
│   │   │   ├── base.py            EnvProtocol 协议定义
│   │   │   └── gymnasium_adapter.py  Gymnasium 环境适配器（含状态保存/恢复）
│   │   ├── faults/
│   │   │   └── base.py            FaultOracle 抽象基类
│   │   ├── abstraction/
│   │   │   └── qvalue_abstraction.py  Q 值离散化抽象策略
│   │   └── ml/
│   │       ├── encoder.py         Episode → 抽象状态二值向量
│   │       └── predictor.py       RandomForest 故障概率预测模型
│   └── Examples/
│       └── cartpole/
│           ├── run.py             SB3 DQN 的框架使用示例
│           └── fault_oracle.py    CartPole 故障判定实现
├── DI-engine/                     DI-engine 框架（作为 DreamerV3 训练后端）
├── experiments/
│   └── dreamerv3_rq1/
│       ├── train_dreamerv3.py     训练并保存 DreamerV3（DI-engine）
│       ├── run_rq1.py             DreamerV3 进入 STARLA 框架的 RQ1 实验入口
│       ├── adapters/
│       │   └── value_abstraction.py  基于 DreamerV3 value 的实验性抽象策略
│       ├── checkpoints/           训练产物（policy、world model、meta）
│       └── results/               实验结果（results.json、RQ1_report.md）
└── MBRL-flat-minima/              其他 MBRL 相关探索
```

---

## 第一步：把 STARLA 改造成可复用测试框架

原始 STARLA 是一个面向 CartPole/MountainCar 的研究 Notebook，核心算法（MOSA 搜索、遗传算子、ML 预测）都混在一个文件里，智能体和环境是写死的。

我做的事情是把它拆开、定义协议、让各层可以独立替换：

### 定义协议层

**智能体协议** `starla/agents/base.py`：
```python
class AgentProtocol(Protocol):
    def predict(self, obs, deterministic=True) -> tuple[Any, dict]: ...
    def get_action_probabilities(self, obs) -> np.ndarray: ...
    def get_q_values(self, obs) -> np.ndarray: ...
```

**环境协议** `starla/envs/base.py`：
```python
class EnvProtocol(Protocol):
    def reset(self) -> Any: ...
    def step(self, action) -> tuple: ...
    def get_state(self) -> Any: ...   # 供 re-execute 保存状态用
    def set_state(self, state) -> Any: ...
```

**故障判定协议** `starla/faults/base.py`：
```python
class FaultOracle(ABC):
    def is_functional_fault(self, episode) -> bool: ...
    def is_reward_fault(self, episode) -> bool: ...
```

### 统一执行入口

`StarlaRunner` 把整个流程分成两步：
1. `prepare_data(training_episodes, random_episodes)`：构建 Q 值抽象、二值编码、训练 RF 故障预测模型、生成初始种群。
2. `run()`：调用 `MOSAEngine` 执行遗传搜索，返回 `TestReport`（含故障统计、archive、计时）。

任何智能体只要实现 `AgentProtocol`，任何环境只要实现 `EnvProtocol`，就能直接接进来测试。

### 已有适配器

- `SB3Agent`：适配 Stable-Baselines3 的 DQN/PPO 等。
- `GymnasiumEnv`：适配所有 Gymnasium 环境，实现了 `get_state/set_state` 用于 re-execute。

---

## 第二步：接入 DI-engine DreamerV3

### 用 DI-engine 训练 DreamerV3

文件：`experiments/dreamerv3_rq1/train_dreamerv3.py`

我用 DI-engine 的 `mbrl_entry_setup` + `DreamerPolicy` 训练 DreamerV3，按环境可用性自动选择后端：
- 优先 `minigrid` / `dmc2gym`
- 不可用时回退到 `CartPole-v1`（`cartpole_fallback`）

训练完成后保存：
- `dreamerv3_policy.pth`
- `dreamerv3_world_model.pth`
- `dreamerv3_meta.json`（记录 backend、exp_name 等元信息）

### 实现 DreamerV3 适配器

文件：`STARLA/src/starla/adapters/dreamerv3/agent.py`

DreamerV3 是一个基于 world model 的序列决策智能体，它的内部结构与 SB3 完全不同，需要特殊处理：

- **`predict`**：维护序列 latent state（`_latent_state`），每步调用 `obs_step + actor` 决策，支持 `reset_state()` 重置。
- **`get_action_probabilities`**：从 actor 分布的 logits/probs 直接提取；连续动作空间退化为均匀分布占位。
- **`get_q_values`**：对每个离散动作做 imagined rollout，用 value head 估出伪 Q 向量——这是 STARLA 中 Q 值抽象（`QValueAbstraction`）能用到 DreamerV3 的核心实现。

---

## 第三步：端到端测试实验（RQ1）

文件：`experiments/dreamerv3_rq1/run_rq1.py`

整个实验链路如下：

```
加载 DreamerV3 checkpoint
        ↓
包装为 DreamerV3Agent（实现 AgentProtocol）
        ↓
用 agent 采样 6 条确定性 episode（训练表现）+ 6 条随机 episode
        ↓
StarlaRunner.prepare_data(...)
  ├── Q 值抽象：对每个状态调用 get_q_values，离散化为抽象类 ID
  ├── EpisodeEncoder：抽象状态集合 → 二值向量
  └── FaultPredictor（RandomForest）：在 training+random 集上训练故障概率模型
        ↓
StarlaRunner.run() → MOSAEngine.run(initial_population)
  ├── 三目标评估：reward fitness / confidence fitness / ML 故障概率
  ├── Crossover：在同一抽象类的状态处拼接两条 episode
  ├── Mutate：在随机时间步加乘性噪声，触发 re-execute 重放
  └── Archive 更新：按目标阈值筛选非支配候选
        ↓
统计 STARLA 发现的功能故障数 / 奖励故障数
        ↓
同等预算下运行纯随机测试作为对照基线
        ↓
输出 results/results.json 与 results/RQ1_report.md
```

### 我在 `run_rq1.py` 里做的关键工程处理

- **兜底 fault oracle**：CartPole 用专用 oracle，其他环境自动用 `GenericGymFaultOracle`（基于 episode 长度和 reward 阈值）。
- **捕获 mutation 次数**：用 monkey patch 包装 `MOSAEngine.run`，捕获 `SearchResult.mutation_count`，用于精确计算测试预算。
- **过短 episode 防护**：包装 `mutate`，对不足 7 步的 episode 跳过变异，避免切片越界崩溃。
- **DreamerV3 heads 兼容**：加载完 world model 后，如果 `heads` 不支持 `.get()` 方法，自动 patch 上——DI-engine 不同版本的 model heads 数据结构不一致导致的问题。

---

## 如何运行

```bash
# 第一步：训练 DreamerV3
python experiments/dreamerv3_rq1/train_dreamerv3.py

# 第二步：运行 RQ1 实验（STARLA vs 随机测试）
python experiments/dreamerv3_rq1/run_rq1.py
```

关键依赖：
```bash
pip install -e STARLA/         # 安装 STARLA 框架（包含 stable-baselines3、gymnasium、sklearn 等）
pip install easydict transformers tensorboardX   # DI-engine 运行时依赖
```

---

## 当前实验结果

环境：`CartPole-v1`，seed=42，budget=1 episode

| 方法 | 功能故障 | 奖励故障 |
|------|---------|---------|
| STARLA | 0 | 1 |
| Random | 0 | 1 |

这是最小规模参数下的一次探索性运行（population=6，generation=3，time budget=60s）。当前结论是两者持平，后续需要扩大参数规模、多种子重复运行才能做统计比较。

---

## 下一步

- 增大 STARLA 搜索规模（population、generations）并多种子重复
- 替换为更有表达力的 latent space 抽象（用 DreamerV3 的 RSSM 隐变量代替伪 Q 值）
- 扩展到更多环境（MiniGrid、DMControl）
- 完整回答 RQ1 / RQ2 / RQ3
