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
│       ├── run_rq1.py             Path A：STARLA 随机变异实验入口
│       ├── adapters/
│       │   └── value_abstraction.py  基于 DreamerV3 value 的实验性抽象策略
│       ├── checkpoints/           训练产物（policy、world model、meta）
│       └── results/               Path A 结果（results.json、RQ1_report.md）
├── experiments_sam/
│   ├── run_sam_experiment.py      Path B：STARLA + SAM 引导变异实验入口
│   ├── adapters/
│   │   └── sam_mutation.py        SAMGuidedMutator（梯度引导变异算子）
│   ├── checkpoints/               复用 dreamerv3_rq1/checkpoints（可为空）
│   └── results/                   Path B 结果（results.json）
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

## 第三步：端到端测试实验（Path A + Path B）

这一阶段我把同一套 STARLA 测试流程拆成两条可直接对比的实验路径：

- **Path A（RQ1 基线）**：保留 STARLA 原始随机变异（随机乘性噪声）
  - 入口：`experiments/dreamerv3_rq1/run_rq1.py`
- **Path B（SAM 引导）**：将随机变异替换为梯度引导变异
  - 入口：`experiments_sam/run_sam_experiment.py`
  - 变异算子：`experiments_sam/adapters/sam_mutation.py`

### 统一主流程（两条路径共享）

```
加载 DreamerV3 checkpoint
        ↓
包装为 DreamerV3Agent（实现 AgentProtocol）
        ↓
采样训练/随机 episode 作为 STARLA 初始数据
        ↓
StarlaRunner.prepare_data(...)
  ├── Q 值抽象（get_q_values -> 离散类 ID）
  ├── EpisodeEncoder（二值向量）
  └── FaultPredictor（RandomForest）
        ↓
StarlaRunner.run() -> MOSAEngine.run(initial_population)
  ├── 三目标评估：reward / confidence / ML fault probability
  ├── crossover
  ├── mutate（Path A 与 Path B 在这里分叉）
  └── archive 更新
        ↓
统计 STARLA 故障 + 用同等预算运行纯随机测试
        ↓
输出 results.json
```

### 两条路径的唯一区别：变异算子

#### Path A：随机变异（RQ1）

在 `starla.core.genetic.transform` 中对状态第一维做随机乘性噪声，属于无方向先验的扰动。

#### Path B：SAM 梯度引导变异

借鉴 SAM 的 `first_step` 思想，将扰动从参数空间迁移到状态空间：

- 参数空间：`w -> w + rho * grad / ||grad||`
- 状态空间：`s -> s + rho * grad_s(-value) / ||grad_s(-value)||`

在实现上，`SAMGuidedMutator` 通过 DreamerV3 world model 的可微链路计算 `grad_s(-value)`，按 L2 归一化后生成扰动；若梯度异常或过小则自动 fallback 到原随机变异，并记录 `fallback_count`。

### 关键工程处理（保证可运行与可比）

- **DreamerV3 heads 兼容补丁**：`world_model.heads` 缺少 `.get()` 时运行时补齐。
- **短 episode 防护**：`len(episode) < 7` 时跳过 mutate，避免随机区间越界。
- **预算可比性**：通过捕获 `SearchResult.mutation_count`，使用 `mutation_count + archive_size` 作为随机对照预算。
- **Path B 无侵入接入**：运行时 monkey patch `genetic.transform`，并在 `finally` 恢复原引用，不修改 STARLA 核心源码。

### 当前保留结果（最小规模验证）

环境：`CartPole-v1`，seed=42，population=6，generation=3，time budget=60s

| 方法 | 功能故障 | 奖励故障 | 备注 |
|------|---------|---------|------|
| Path A：STARLA 随机变异 | 0 | 1 | `experiments/dreamerv3_rq1/results/results.json` |
| Path B：STARLA + SAM 变异 | 0 | 1 | `experiments_sam/results/results.json`（`sam_fallback_count=0`） |
| 纯随机测试（同预算） | 0 | 1 | 来自各自 `random_*` 字段 |

当前阶段结论：两条路径在最小规模设置下持平，说明 Path B 链路已正确打通，但还需要更大预算与多 seed 才能判断统计意义上的优劣。

---

## 如何运行

```bash
# 第一步：训练 DreamerV3
python experiments/dreamerv3_rq1/train_dreamerv3.py

# 第二步A：运行 Path A（STARLA 随机变异）
python experiments/dreamerv3_rq1/run_rq1.py

# 第二步B：运行 Path B（STARLA + SAM 引导变异）
python experiments_sam/run_sam_experiment.py
```

关键依赖：
```bash
pip install -e STARLA/         # 安装 STARLA 框架（包含 stable-baselines3、gymnasium、sklearn 等）
pip install easydict transformers tensorboardX   # DI-engine 运行时依赖
```

---

## 下一步

- 增大 STARLA 搜索规模（population、generations）并多种子重复
- 替换为更有表达力的 latent space 抽象（用 DreamerV3 的 RSSM 隐变量代替伪 Q 值）
- 扩展到更多环境（MiniGrid、DMControl）
- 完整回答 RQ1 / RQ2 / RQ3
