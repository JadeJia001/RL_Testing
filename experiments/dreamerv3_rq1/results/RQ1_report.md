## 1. 实验设置

- 环境名称：`CartPole-v1`
- DreamerV3 训练步数：`max_env_step = 10000`；随机种子：`42`
- STARLA 搜索参数：`population_size = 6`，`num_generations = 3`，`time_budget_seconds = 60`
- 预算定义（Scenario 1）：`STARLA 预算 = 搜索期间变异执行数 + 最终 archive 执行数`

## 2. 结果对比

| 方法 | 测试预算(episode数) | 功能故障数 | 奖励故障数 |
|------|---------------------|-----------|-----------|
| STARLA | 1 | 0 | 1 |
| Random Testing | 1 | 0 | 1 |

## 3. RQ1 回答

在同等测试预算（1 个 episode）下，STARLA 与 Random Testing 发现的故障数量相同（功能故障均为 0，奖励故障均为 1），因此本次实验中 STARLA **没有**发现更多故障。

## 4. 局限性

- 当前为最小规模复现实验，参数较小，不做统计显著性检验。
- DreamerV3 的伪 Q 值抽象可能影响搜索效率，后续可改进为 latent-space abstraction。
- 后续建议：调大参数、多种子多次运行取平均。
