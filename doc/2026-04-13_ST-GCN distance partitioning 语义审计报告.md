# ST-GCN `distance partitioning` 语义审计报告

## 1. 审计背景

在 NTU RGB+D xsub 第一轮消融中，4 组实验的总体趋势已经比较清楚：

- `spatial_imp` 最强
- `spatial_noimp` 次之
- `uniform_noimp` 明显更弱
- `distance_noimp` 最异常，验证集表现强烈震荡

已知事实是：

- `uniform / spatial / spatial+imp` 的训练与验证曲线都基本正常
- 只有 `distance_noimp` 出现“训练能收敛，但验证大幅塌陷又恢复”的现象
- 基础结构检查已确认邻接矩阵 shape 正常：
  - `uniform = (1, 25, 25)`
  - `distance = (2, 25, 25)`
  - `spatial = (3, 25, 25)`

因此这次审计的目标不是继续做 shape 层面的检查，而是直接检查 `distance partitioning` 的语义是否符合论文意图，重点看：

1. 子集切分是否纯净
2. 切分前后 support 和数值是否守恒
3. 归一化顺序是否合理
4. toy graph 上的行为是否符合直觉
5. 最小前向下 `uniform` 与 `distance` 是否存在异常数值偏置

## 2. 审计对象

本次主要审计以下实现：

- [net/utils/graph.py](./net/utils/graph.py)
- [net/utils/tgcn.py](./net/utils/tgcn.py)

另外补了一个可重复执行的审计脚本：

- [temp/audit_distance_partition.py](./temp/audit_distance_partition.py)

## 3. 子集语义检查

### 3.1 论文语义

在 `D = 1` 时，`distance partitioning` 应只有两个子集：

1. `d = 0`：root node itself
2. `d = 1`：all other 1-hop neighbors

### 3.2 当前实现检查结果

对当前 NTU 图的 `distance` 策略直接检查后，结果如下：

- `A[0]` 只包含对角 self connections：`True`
- `A[1]` 包含 self-loop：`False`
- `A[1]` 只包含 1-hop non-self edges：`True`
- `A[1]` 混入 2-hop 或更远边：`False`

因此，从“集合切分是否干净”这个层面看，当前实现没有问题。

## 4. 守恒性检查

进一步检查：

- `support(A_uniform) == support(A_distance[0] + A_distance[1])`：`True`
- `A_uniform` 与 `A_distance[0] + A_distance[1]` 的逐元素最大差异：`0.0`

这说明：

- `distance` 不是漏边或多边
- 它只是把 `uniform` 的总邻接拆成两个互斥子集
- 从 support 和合并结果看，当前 `distance` 是严格守恒的

## 5. 归一化顺序审计

### 5.1 当前实现

当前实现位于 [net/utils/graph.py](./net/utils/graph.py)，核心顺序是：

1. 先构造 `hop <= max_hop` 的整体邻接矩阵
2. 对整体邻接矩阵做一次 `normalize_digraph`
3. 再按 `hop == 0 / hop == 1` 切分到 `distance` 的两个子集

也就是说，当前实现是：

> **先整体归一化，再切分**

而不是：

> **先切分，再对每个 subset 单独归一化**

### 5.2 为什么这里最可疑

这一步是本次审计中最值得怀疑的位置。

原因是：

- 如果先整体归一化，那么 `distance[0]` 和 `distance[1]` 拿到的只是“整体邻接矩阵归一化后的残片”
- 两个 subset 的权重会共享同一套 degree 归一化口径
- 这会削弱“subset 作为独立关系子图参与卷积”的语义

如果从论文中“按 subset 拆分邻居关系”的直观语义理解，更合理的做法通常应该是：

- 先切成 `d=0` 和 `d=1`
- 再对子集分别归一化

## 6. Toy Graph 验证

为了把这个差异看清楚，本次用一个 3 节点链图 `0-1-2` 做了最小 toy test。

### 6.1 当前实现：先整体归一化，再切分

`distance[0]`：

```text
[[0.5    0.     0.    ]
 [0.     0.3333 0.    ]
 [0.     0.     0.5   ]]
```

`distance[1]`：

```text
[[0.     0.3333 0.    ]
 [0.5    0.     0.5   ]
 [0.     0.3333 0.    ]]
```

### 6.2 另一种口径：先切分，再对子集单独归一化

`distance[0]`：

```text
[[1. 0. 0.]
 [0. 1. 0.]
 [0. 0. 1.]]
```

`distance[1]`：

```text
[[0.  0.5 0. ]
 [1.  0.  1. ]
 [0.  0.5 0. ]]
```

### 6.3 审计结论

这两种结果差异非常明显。

当前实现下：

- self subset 被整体 degree 稀释
- hop-1 subset 也被整体 degree 稀释

因此，`distance` 的两个 subset 并没有以“独立子图”的方式被平衡，而只是从整体归一化邻接矩阵中切出两个部分。

这正是本次审计里最可疑的地方。

对应可视化如下。上排是当前实现，下排是 `distance_subsetnorm` 口径；最右一列给出两组子集合并后的总邻接，可以直接看到当前实现下两个 subset 都被整体 degree 一起稀释，而 `subsetnorm` 则把 self 与 hop-1 都抬成了各自独立归一化后的满权重分支。

![Toy graph 下两种 `distance` 归一化口径的矩阵对比](./work_dir/figures/ntu_xsub_distance_semantic_toy_matrices.png)

## 7. 最小前向测试

本次还做了一个最小前向一致性检查：

- `uniform` 使用 `K=1`
- `distance` 使用 `K=2`
- 强行把 `distance` 两个 subset 的卷积权重设成完全相同
- 输入同一个随机张量
- 比较 `uniform` 和 `distance` 的输出

结果：

- `uniform mean/std = -0.01619 / 0.31495`
- `distance mean/std = -0.01619 / 0.31495`
- `max abs diff = 5.96e-08`
- `allclose = True`

这说明：

- 当前 `distance` 前向本身没有额外的数值偏置
- 只要两个 subset 的权重相同，它与 `uniform` 就是严格等价的
- 因此问题不在 [net/utils/tgcn.py](./net/utils/tgcn.py) 的前向聚合公式本身

这个结论也可以直接从受控前向图里看到：左图采样点几乎完全压在 `y=x` 上，右图的绝对误差分布则收缩在数值精度噪声附近，说明 `distance` 没有因为“多一个 subset”就在前向聚合里引入额外偏置。

![`uniform` 与 `distance` 在受控前向下的输出一致性](./work_dir/figures/ntu_xsub_distance_semantic_forward_equivalence.png)

## 8. 三类结论

### 8.1 已排除

以下问题当前已经基本排除：

- `distance[0] / distance[1]` 子集切分错误
- `A[1]` 混入 self-loop
- `A[1]` 混入 2-hop edge
- `distance` support 不守恒
- `ConvTemporalGraphical` 前向对 `distance` 存在额外数值偏置

### 8.2 仍可疑

当前仍然可疑的是：

- `distance` 的异常更像是“切分语义正确，但归一化语义不够独立”
- `distance + no edge importance` 在当前归一化口径下，比 `uniform` 更复杂，却又不具备 `spatial` 那么强的结构先验
- 这会让它表现成“训练 loss 能降，但验证端强烈震荡”

### 8.3 高优先级修复点

当前最高优先级的修复点就是：

- [net/utils/graph.py](./net/utils/graph.py) 中 `distance` 分支的归一化顺序

建议最小修复方向：

- 不覆盖当前 `distance` 语义
- 新增一个“先切分、再对子集独立归一化”的 `distance` 变体
- 用最小短程验证直接比较其与当前 `distance` 的稳定性差异

## 9. 初步结论

基于这次语义审计，目前最稳妥的判断是：

- `distance_noimp` 的异常不太像简单实现 bug
- 真正最值得怀疑的是 `distance partitioning` 当前采用了“整体归一化后再切分”的语义
- 这可能使 `distance` 子集的贡献没有按 subset 独立平衡，进而放大了验证端不稳定

因此，下一步最合理的动作不是继续做更多猜测，而是直接实现一个“per-subset normalization”的最小修复变体，并做短程对照验证。

## 10. 最小修复探针

### 10.1 修复策略

为了避免直接改写原有 `distance` 语义，本次没有覆盖既有策略，而是在
[net/utils/graph.py](./net/utils/graph.py) 中新增了一个最小探针策略：

- `distance_subsetnorm`

其逻辑是：

1. 先按 `hop==0 / hop==1` 切出两个 subset
2. 再对子集分别调用 `normalize_digraph`

也就是把原来的“整体归一化后再切分”改成了“先切分、再对子集独立归一化”。

对应探针配置为：

- [config/ablation/ntu-xsub/distance_subsetnorm_probe.yaml](./config/ablation/ntu-xsub/distance_subsetnorm_probe.yaml)

该配置保持与第一轮消融主协议尽量一致，只把训练长度缩短到 `15 epoch`，用于快速判断修复方向是否值得继续。

### 10.2 探针结果

为便于比较，先列出原始 `distance_noimp` 在对应阶段的验证结果：

- `epoch 4`: `Top1=55.61%`, `Top5=88.25%`
- `epoch 9`: `Top1=41.18%`, `Top5=74.59%`
- `epoch 14`: `Top1=43.96%`, `Top5=78.32%`

`distance_subsetnorm` 探针对应结果为：

- `epoch 4`: `Top1=52.06%`, `Top5=82.77%`
- `epoch 9`: `Top1=60.08%`, `Top5=88.82%`
- `epoch 14`: `Top1=60.18%`, `Top5=89.35%`

### 10.3 探针读法

这组探针呈现出两个同时成立的现象：

- 在首个验证点，`distance_subsetnorm` 低于原始 `distance_noimp`
- 在中期验证点，`distance_subsetnorm` 明显高于原始 `distance_noimp`

因此，这个探针支持如下判断：

- “先切分、再对子集独立归一化”确实触及了异常来源
- 但它不是一个“改完后整体单调变好”的修复
- 更准确的描述是：它改变了 `distance` 的数值平衡方式，使模型前期更激进，但中期明显更稳

把短程探针对照画出来后，这个现象会更直观：上面三张子图保留了原始 `distance_noimp` 的完整 `40 epoch` 轨迹，灰色区域表示“原始基线还在继续，但 probe 已经在 `epoch 14` 停止”；右下角的差值子图只比较两组都真正出现过的公共评估点。因此这里不是“图只画了 15 epoch”，而是明确把“probe 只跑到 15 epoch”这件事编码进图里，同时保留基线后续的异常震荡背景。

![`distance_noimp` 与 `distance_subsetnorm_probe` 的短程验证曲线对比](./work_dir/figures/ntu_xsub_distance_semantic_probe_curves.png)

## 11. 为什么首点更差，但中期更稳

这部分不再讨论“修复是否有效”，而只回答一个更具体的问题：

为什么 `distance_subsetnorm` 会表现成“首点更差，但中期更稳”。

### 11.1 权重语义变化

对 NTU 图直接检查后，原始 `distance` 与 `distance_subsetnorm` 的列归一化口径差异如下：

- 原始 `distance`
  - self subset 的列和范围：`0.2 ~ 0.5`
  - hop-1 subset 的列和范围：`0.5 ~ 0.8`
  - 两个 subset 合并后每列总和恒为 `1`
- `distance_subsetnorm`
  - self subset 的列和恒为 `1`
  - hop-1 subset 的列和也恒为 `1`
  - 两个 subset 合并后每列总和恒为 `2`

这意味着 `distance_subsetnorm` 做了两件事：

1. 把 self 分支从“弱分支”直接抬成了“满权重分支”
2. 把整个图卷积的总消息量翻倍

换句话说，这个变体不是小修小补，而是把 `distance` 的数值语义从：

- 邻居主导、self 较弱

改成了：

- self / neighbor 两个分支同强

如果只看这段文字，容易停留在抽象描述；把 NTU 图每个节点上的列和直接展开后就更容易理解。当前实现里，self 列和稳定落在 `0.2 ~ 0.5`，neighbor 落在 `0.5 ~ 0.8`，合并后每列守恒为 `1`；而 `distance_subsetnorm` 则把 self 与 neighbor 两条曲线都抬成了恒定 `1`，合并后每列总权重直接翻到 `2`。

![NTU 图上两种归一化顺序的 self/neighbor 列和分布对比](./work_dir/figures/ntu_xsub_distance_semantic_column_balance.png)

### 11.2 同权重前向下的尺度变化

为了排除随机初始化噪声，使用完全相同的卷积权重，只替换邻接矩阵 `A` 做了受控前向对比。

结果表明：

- 在最小 `ConvTemporalGraphical` 测试里，`distance_subsetnorm` 的输出绝对均值约为原始 `distance` 的 `2.05x`
- 在整模型 10 个 ST-GCN block 的逐层对比里，`distance_subsetnorm` 的激活绝对均值长期维持在原始 `distance` 的 `1.5x ~ 2.0x`
- 最终 logits 的绝对均值仍约高 `13%`

因此，这个变体并不是“和原始版差不多，只是归一化口径更干净”，而是从第一层开始就持续产生更强的激活。

对应的受控尺度图把这件事拆成了三个层次：

- 左上：最小 `ConvTemporalGraphical` 输出绝对均值直接放大到约 `2.05x`
- 右上：10 个 ST-GCN block 的激活绝对均值几乎全程高于原始 `distance`
- 下排：最终 logits 绝对均值仍保持更高，且 block-wise ratio 长期落在 `1.5x ~ 2.0x`

![`distance_subsetnorm` 在受控前向下的尺度放大对比](./work_dir/figures/ntu_xsub_distance_semantic_scale_amplification.png)

### 11.3 对训练曲线的解释

这与实际日志是对得上的。

在首个验证点：

- 原始 `distance_noimp`
  - train mean loss: `1.3344`
  - val mean loss: `1.5824`
  - Top1: `55.61`
- `distance_subsetnorm`
  - train mean loss: `1.4656`
  - val mean loss: `1.8628`
  - Top1: `52.06`

这说明该变体在前期确实更难训、也更不保守；首点变差是稳定现象，不是偶然噪声。

但在中期：

- 原始 `distance_noimp`
  - `epoch 9 / 14` 的 val mean loss: `2.5354 / 2.4670`
  - Top1: `41.18 / 43.96`
- `distance_subsetnorm`
  - `epoch 9 / 14` 的 val mean loss: `1.4316 / 1.4140`
  - Top1: `60.08 / 60.18`

这说明原始 `distance` 的真正问题不是“学不会”，而是：

- 训练集 loss 继续下降
- 验证集 loss 却在中期剧烈恶化

而 `distance_subsetnorm` 至少显著抑制了这种中期塌陷。

### 11.4 当前最合理的解释

基于现有证据，当前更稳定的解释是：

- 原始 `distance` 由于 self 分支过弱、neighbor 分支相对过强，更像一种“邻居主导”的聚合
- 这种口径在训练前期更保守，所以首个验证点反而略好
- 但随着训练推进，它更容易出现身份信息保留不足、邻域混合过强或泛化塌陷的问题
- `distance_subsetnorm` 则把 self 分支显著抬高，虽然前期更激进、首点更差，但中期不容易出现同样的验证崩塌

因此，“首点更差但中期更稳”并不矛盾，它反映的是：

- 这个变体改变了 `distance` 的数值平衡方式
- 这个改变牺牲了一部分前期保守性
- 换来了中期明显更稳定的泛化表现

## 12. 当前结论

把结构审计、toy graph、受控前向和短程探针合在一起，当前更适合长期保留的结论是：

- `distance partitioning` 的子集切分本身是干净的，support 也守恒，问题不在“边切错了”
- `ConvTemporalGraphical` 的前向聚合公式没有单独对 `distance` 引入异常偏置
- 当前实现中最可疑的地方仍是 [net/utils/graph.py](./net/utils/graph.py) 里原始 `distance` 的“先整体归一化，再切分”
- 但把它改成“先切分、再对子集独立归一化”后，带来的不是单一收益，而是一次明显的数值语义变化
- 这次变化的直接效果是：前期更激进，中期更稳定
- 因此，`distance_noimp` 的异常不能只被描述为“一个归一化顺序 bug”，更准确的说法应是：
  - 原始 `distance` 的 subset 平衡方式不理想
  - 但真正要做成稳定、可解释的修复，还需要继续约束总消息量与 self / neighbor 的相对权重

## 13. 补充图像产物

本次为语义审计报告补充了 3 张直接服务于结论阅读的对比图：

- Toy graph 归一化语义对比：
  [work_dir/figures/ntu_xsub_distance_semantic_toy_matrices.png](./work_dir/figures/ntu_xsub_distance_semantic_toy_matrices.png)
  /
  [work_dir/figures/ntu_xsub_distance_semantic_toy_matrices.pdf](./work_dir/figures/ntu_xsub_distance_semantic_toy_matrices.pdf)
- 短程探针对照曲线：
  [work_dir/figures/ntu_xsub_distance_semantic_probe_curves.png](./work_dir/figures/ntu_xsub_distance_semantic_probe_curves.png)
  /
  [work_dir/figures/ntu_xsub_distance_semantic_probe_curves.pdf](./work_dir/figures/ntu_xsub_distance_semantic_probe_curves.pdf)
- NTU 图列和平衡对比：
  [work_dir/figures/ntu_xsub_distance_semantic_column_balance.png](./work_dir/figures/ntu_xsub_distance_semantic_column_balance.png)
  /
  [work_dir/figures/ntu_xsub_distance_semantic_column_balance.pdf](./work_dir/figures/ntu_xsub_distance_semantic_column_balance.pdf)
- 最小前向一致性对比：
  [work_dir/figures/ntu_xsub_distance_semantic_forward_equivalence.png](./work_dir/figures/ntu_xsub_distance_semantic_forward_equivalence.png)
  /
  [work_dir/figures/ntu_xsub_distance_semantic_forward_equivalence.pdf](./work_dir/figures/ntu_xsub_distance_semantic_forward_equivalence.pdf)
- 受控尺度放大对比：
  [work_dir/figures/ntu_xsub_distance_semantic_scale_amplification.png](./work_dir/figures/ntu_xsub_distance_semantic_scale_amplification.png)
  /
  [work_dir/figures/ntu_xsub_distance_semantic_scale_amplification.pdf](./work_dir/figures/ntu_xsub_distance_semantic_scale_amplification.pdf)

对应脚本为：

- [tools/plot_distance_semantic_audit.py](./tools/plot_distance_semantic_audit.py)
