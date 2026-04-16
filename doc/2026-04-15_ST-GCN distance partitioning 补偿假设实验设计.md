# ST-GCN `distance partitioning` 补偿假设实验设计

## 1. 文档目的

本文档用于固定第二轮 `distance partitioning` 相关实验的目标、假设、实现口径和执行顺序，避免后续在实现过程中不断漂移问题定义。

这轮实验**不是**重新铺一整套大消融，而是围绕已经暴露出的 `distance_noimp` 异常，验证两个更具体的问题：

1. `edge importance weighting` 是否在补偿原版 `distance` 的 subset balance 偏置。
2. 在控制总消息量公平的前提下，`subset-wise normalization` 是否仍然带来独立于 `importance` 的净收益。

## 2. 背景

第一轮消融已经得到以下稳定事实：

- `spatial_imp` 最强
- `spatial_noimp` 次之
- `uniform_noimp` 明显更弱
- `distance_noimp` 最异常，验证曲线强烈震荡

已有语义审计进一步表明：

- `distance` 的子集切分本身是干净的
- `support(A_uniform) == support(A_distance[0] + A_distance[1])`
- `ConvTemporalGraphical` 前向本身没有对 `distance` 引入额外数值偏置
- 当前最可疑的位置是 [net/utils/graph.py](../net/utils/graph.py) 中原始 `distance` 的“先整体归一化，再切分”

已有最小探针还说明：

- 裸 `distance_subsetnorm` 能明显缓解中期验证塌陷
- 但它会把总消息量放大，导致首个验证点更差

因此，这轮实验不再使用裸 `subsetnorm`，而是采用 **`subsetnorm_rescaled`** 口径。

## 3. 核心假设

### H1：原版 `distance` 的主要问题是 subset balance 不理想

当前异常更像是：

- 子集切分语义正确
- 但 `self` / `neighbor` 两个 subset 的相对强度不理想

而不是：

- 边切错了
- support 不守恒
- 图卷积公式错了

### H2：`importance` 可能在补偿这种不理想的 balance

如果这个假设成立，那么：

- `distance_imp` 应显著优于 `distance_noimp`
- 原版 `distance_imp` 的有效邻接 `A_eff = A ⊙ M` 应表现出对某个 subset 的系统性补偿
- `subsetnorm_rescaled_imp` 与 `distance_imp` 的差距，可以反映这种补偿是否已经足够

### H3：`subsetnorm_rescaled` 可能带来独立于 `importance` 的净收益

如果这个假设成立，那么：

- `distance_subsetnorm_rescaled_noimp` 应比原始 `distance_noimp` 更稳
- `distance_subsetnorm_rescaled_imp` 应比 `distance_imp` 进一步受益

## 4. `subsetnorm_rescaled` 的固定定义

这轮实验里，`subsetnorm_rescaled` 的定义必须固定，不允许边做边改。

### 4.1 目标

该口径用于同时满足两件事：

1. 保留 “先切分、再对子集独立归一化” 的论文语义
2. 避免裸 `subsetnorm` 把总消息量整体放大

### 4.2 定义

记共有 `K` 个 subset。

实现步骤固定为：

1. 先按 subset 切分邻接矩阵
2. 对每个 subset 单独调用 `normalize_digraph`
3. 再将每个 subset 的结果统一乘以 `1 / K`

也就是：

\[
\tilde{A}_j = \frac{1}{K} \cdot \mathrm{Normalize}(A_j)
\]

### 4.3 适用策略

本轮仅对以下策略引入该口径：

- `distance_subsetnorm_rescaled`
- `spatial_subsetnorm_rescaled`

其中：

- `distance` 的 `K=2`
- `spatial` 的 `K=3`

### 4.4 不做的变体

为控制变量，本轮不做：

- 裸 `subsetnorm`
- 非 `1/K` 的其他 rescale 方案
- `uniform` 的任何新变体

## 5. 实验协议

### 5.1 第一阶段：20 epoch 筛查

用于快速判断方向，不追最终最好点。

固定实验：

- `A1 = distance_imp`
- `A2 = distance_subsetnorm_rescaled_noimp`
- `A3 = distance_subsetnorm_rescaled_imp`
- `A4 = spatial_subsetnorm_rescaled_noimp`

### 5.2 第二阶段：40 epoch 确认

用于确认第一阶段最有信息量的分支。

固定保留：

- `B1 = distance_imp`

第二个按第一阶段结果二选一：

- 若 `A3` 在中期持续优于 `A1`，则 `B2 = distance_subsetnorm_rescaled_imp`
- 否则 `B2 = spatial_subsetnorm_rescaled_noimp`

## 6. 统一训练协议

除非后续单独开文档修订，本轮实验统一使用以下协议：

- 数据集：`NTU RGB+D xsub`
- 精度：`AMP on`
- seed：固定为 `49`
- `batch_size = 16`
- `test_batch_size = 16`
- `base_lr = 0.025`
- `optimizer = SGD`
- `weight_decay = 0.0001`
- `nesterov = True`
- `eval_interval = 5`

### 6.1 epoch 与 step

第一阶段：

- `num_epoch = 20`
- `step = [10]`

第二阶段：

- `num_epoch = 40`
- `step = [10, 20, 30]`

这样做的目的不是拟合论文终版 schedule，而是让 20 epoch 与 40 epoch 都有内部一致的短程协议。

## 7. 第一阶段的判据

20 epoch 不以最终绝对精度为主，重点看以下三类信号。

### 7.1 稳定性

重点观察：

- `epoch 4 / 9 / 14 / 19` 的 `Top-1`
- 同节点的 `val mean_loss`
- 是否出现类似原始 `distance_noimp` 的中期塌陷

### 7.2 相对提升

固定比较以下对：

- `A1` vs 原始 `distance_noimp`
- `A2` vs 原始 `distance_noimp`
- `A3` vs `A1`
- `A4` vs 原始 `spatial_noimp`

### 7.3 决策规则

第二阶段的 `B2` 选择不使用单一分数阈值，而是综合以下三点：

- 中期节点是否持续占优
- `val mean_loss` 是否持续更低
- 曲线是否更平顺

## 8. 结果解释模板

### 8.1 结论 A：`importance` 已足够补偿

若：

- `distance_imp` 明显优于 `distance_noimp`
- `distance_subsetnorm_rescaled_imp` 不再明显优于 `distance_imp`

则可解释为：

- `distance` 的主要问题更像是缺少可学习边权时的 subset balance 不足
- 加上 `importance` 后，大部分问题已经被模型自行补偿

### 8.2 结论 B：`subsetnorm_rescaled` 仍有独立收益

若：

- `distance_subsetnorm_rescaled_noimp` 比 `distance_noimp` 更稳
- `distance_subsetnorm_rescaled_imp` 又进一步优于 `distance_imp`

则可解释为：

- `importance` 确实在补偿
- 但更公平的 subset 归一化口径仍有独立收益

### 8.3 结论 C：影响主要集中在 `distance`

若：

- `spatial_subsetnorm_rescaled_noimp` 与原始 `spatial_noimp` 差异很小

则更适合解释为：

- 该实现口径问题主要集中在 `distance`
- `spatial` 由于结构先验更强，对同类偏置不那么敏感

## 9. `A_eff` 统计设计

这轮不优先做热图，只做轻量统计。

### 9.1 统计对象

只分析这两个模型：

- `distance_imp`
- `distance_subsetnorm_rescaled_imp`

### 9.2 统计量

对每一层、每个 subset 记录：

- `mask_mean`
  - `M` 的平均值
- `a_eff_sum`
  - `A_eff = A ⊙ M` 的总和
- `a_eff_colsum_mean`
  - `A_eff` 的列和均值
- `a_eff_abs_mean`
  - `A_eff` 绝对值均值

另行计算：

- `self / neighbor` 的 `mask_mean` 比值
- `self / neighbor` 的 `a_eff_colsum_mean` 比值

### 9.3 主要问题

这些统计只服务于一个核心判断：

> 原版 `distance_imp` 是否在系统性地抬高 self subset，来补偿原始实现里 self 偏弱的问题。

## 10. 这轮明确不做的事

为控制总时长和变量边界，本轮不做：

- `uniform` 任何新变体
- 多 seed
- `spatial_subsetnorm_rescaled_imp`
- 大规模热图可视化
- 裸 `subsetnorm`

## 11. 执行清单

### 第 0 步：准备

需要补齐：

- `distance_imp` 配置
- `distance_subsetnorm_rescaled_noimp` 配置
- `distance_subsetnorm_rescaled_imp` 配置
- `spatial_subsetnorm_rescaled_noimp` 配置
- `A_eff` 统计脚本

### 第 1 步：第一阶段

顺序执行：

1. `A1 distance_imp`
2. `A2 distance_subsetnorm_rescaled_noimp`
3. `A3 distance_subsetnorm_rescaled_imp`
4. `A4 spatial_subsetnorm_rescaled_noimp`

### 第 2 步：第二阶段

顺序执行：

1. `B1 distance_imp`
2. `B2` 在 `distance_subsetnorm_rescaled_imp` 与 `spatial_subsetnorm_rescaled_noimp` 中二选一

### 第 3 步：统计与总结

导出 `A_eff` 轻量统计，回答两句话：

1. `importance` 能否把原版 `distance` 的异常大部分学掉。
2. 在总量公平前提下，`subsetnorm_rescaled` 是否仍存在独立于 `importance` 的净收益。

## 12. 对应分析图规划

这轮设计虽然先于结果落地，但分析视角不应等跑完后再临时决定。当前固定 5 张对应图，各自回答不同问题：

- 阶段一筛查图：
  [work_dir/figures/ntu_xsub_distance_compensation_stage1.png](./work_dir/figures/ntu_xsub_distance_compensation_stage1.png)
  用于回答 `A1/A2/A3/A4` 在 `20 epoch` 内谁更稳、谁只是中期修复、谁是结构性领先。
- 阶段二确认图：
  [work_dir/figures/ntu_xsub_distance_compensation_stage2.png](./work_dir/figures/ntu_xsub_distance_compensation_stage2.png)
  用于回答 `B2` 是否在更长训练里继续压住 `B1`，而不是只在短程里占优。
- 与第一轮基线桥接图：
  [work_dir/figures/ntu_xsub_distance_compensation_baseline_bridge.png](./work_dir/figures/ntu_xsub_distance_compensation_baseline_bridge.png)
  用于回答 `distance` 家族是否真的摆脱了原始塌陷，以及 `spatial` 家族是否本来就没有同量级问题。
- 实验结果总图：
  [work_dir/figures/ntu_xsub_distance_compensation_results.png](./work_dir/figures/ntu_xsub_distance_compensation_results.png)
  用于保留全局鸟瞰，不替代分阶段图。
- 权重分析图：
  [work_dir/figures/ntu_xsub_distance_compensation_weights.png](./work_dir/figures/ntu_xsub_distance_compensation_weights.png)
  用于回答 `importance` 学出的到底是 `mask` 平均值差异，还是 `A_eff` 子集强度重心变化。

## 13. 当前接受的工作结论

这份实验设计当前默认接受以下前提：

- `subsetnorm_rescaled` 的目的就是平衡裸 `subsetnorm` 带来的归一化总量增加
- 当前最关键的不是继续验证 shape、support 或 tgcn 前向，而是验证 subset balance 与补偿关系
- 这轮实验是对“补偿假设”的验证，不是对“论文实现唯一正确口径”的最终判定
