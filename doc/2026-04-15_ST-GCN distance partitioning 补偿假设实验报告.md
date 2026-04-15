# ST-GCN `distance partitioning` 补偿假设实验报告

## 1. 报告目的

本报告对应 [2026-04-15_ST-GCN distance partitioning 补偿假设实验设计](./2026-04-15_ST-GCN%20distance%20partitioning%20补偿假设实验设计.md)，用于记录第二轮 `distance partitioning` 相关实验的最终结果、权重统计与结论。

这轮实验不是重新铺一整套大消融，而是围绕一个更具体的问题展开：

1. 原版 `distance` 的异常，是否主要来自 subset balance 偏置。
2. `edge importance weighting` 是否在补偿这种偏置。
3. 在总消息量公平前提下，`subsetnorm_rescaled` 是否仍带来独立于 `importance` 的净收益。

## 2. 对应实验设计与统一协议

### 2.1 阶段一：20 epoch 筛查

本阶段共 4 组实验：

- `A1 = distance_imp`
- `A2 = distance_subsetnorm_rescaled_noimp`
- `A3 = distance_subsetnorm_rescaled_imp`
- `A4 = spatial_subsetnorm_rescaled_noimp`

统一协议如下：

- 数据集：NTU RGB+D xsub
- 训练轮数：`20`
- 评估间隔：`5`
- `batch_size = 16`
- `test_batch_size = 16`
- `base_lr = 0.025`
- `step = [10]`
- `optimizer = SGD`
- `nesterov = True`
- `weight_decay = 0.0001`
- AMP：开启
- seed：固定为 `49`

### 2.2 阶段二：40 epoch 确认

根据阶段一结果，最终拉长到 40 epoch 的 2 组实验为：

- `B1 = distance_imp`
- `B2 = distance_subsetnorm_rescaled_imp`

统一协议如下：

- 训练轮数：`40`
- 学习率衰减：`step = [10, 20, 30]`
- 其余设置与阶段一保持一致

## 3. 六组实验结果

### 3.1 阶段一结果（20 epoch）

| 编号 | 实验 | Best Top-1 | Best Epoch | Final Top-1 | Best Top-5 | Final Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| `A1` | `distance_imp` | `77.12%` | `20` | `77.12%` | `95.81%` | `95.81%` |
| `A2` | `distance_subsetnorm_rescaled_noimp` | `72.71%` | `15` | `72.57%` | `94.53%` | `94.53%` |
| `A3` | `distance_subsetnorm_rescaled_imp` | `77.59%` | `20` | `77.59%` | `95.92%` | `95.92%` |
| `A4` | `spatial_subsetnorm_rescaled_noimp` | `74.00%` | `15` | `73.77%` | `95.11%` | `95.04%` |

如果只看阶段一 4 组内部排序，则结果为：

`A3 > A1 >> A4 > A2`

其中最关键的是：

- `A3` 全程略优于 `A1`
- `A2` 虽然仍弱于 `importance` 版本，但曲线明显比历史 `distance_noimp` 更稳
- `A4` 与历史 `spatial_noimp` 十分接近，没有表现出同量级的异常

### 3.2 阶段二结果（40 epoch）

| 编号 | 实验 | Best Top-1 | Best Epoch | Final Top-1 | Best Top-5 | Final Top-5 |
| --- | --- | --- | --- | --- | --- | --- |
| `B1` | `distance_imp` | `76.31%` | `35` | `76.10%` | `95.49%` | `95.40%` |
| `B2` | `distance_subsetnorm_rescaled_imp` | `77.79%` | `40` | `77.79%` | `96.37%` | `96.26%` |

`B2` 相比 `B1` 的最终收益为：

- `Top-1 +1.69`
- `Top-5 +0.86`

并且从 `epoch 14` 开始，`B2` 的验证 loss 持续低于 `B1`：

- `epoch 14`: `0.7985 < 0.8510`
- `epoch 19`: `0.7912 < 0.8598`
- `epoch 24`: `0.7500 < 0.8333`
- `epoch 29`: `0.7568 < 0.8253`
- `epoch 34`: `0.7487 < 0.8105`
- `epoch 39`: `0.7403 < 0.8224`

## 4. 与原始参考基线的关系

为了避免把这轮实验读成“仅看六组内部排序”，这里补一层与前一轮原始参考基线的对照。下列原始参考值都来自 40 epoch 第一轮消融中对应 epoch 节点：

### 4.1 `distance_noimp` 参考

原始 `distance_noimp` 在前一轮实验中的验证曲线为：

- `epoch 5`: `55.61% / 86.76%`
- `epoch 10`: `41.18% / 76.55%`
- `epoch 15`: `43.96% / 79.08%`
- `epoch 20`: `61.85% / 89.53%`

对照当前阶段一的两组 `distance_subsetnorm_rescaled`：

- `A2`：`55.03 -> 65.58 -> 72.71 -> 72.57`
- `A3`：`61.60 -> 69.38 -> 76.73 -> 77.59`

可以看到：

1. `subsetnorm_rescaled_noimp` 已经明显抑制了原始 `distance_noimp` 的中期塌陷。  
   它在 `epoch 10 / 15 / 20` 的 `Top-1` 分别比原始 `distance_noimp` 高出 `+24.40 / +28.75 / +10.72` 个点。

2. 在此基础上再加入 `importance`，可以把最终 `Top-1` 进一步抬到 `77.59%`。  
   这说明 `importance` 不是唯一因素，但仍然有显著补偿作用。

### 4.2 `spatial_noimp` 参考

原始 `spatial_noimp` 在前一轮实验中对应节点为：

- `epoch 5`: `58.85% / 88.16%`
- `epoch 10`: `67.25% / 92.39%`
- `epoch 15`: `74.33% / 94.72%`
- `epoch 20`: `73.26% / 94.66%`

当前 `A4 = spatial_subsetnorm_rescaled_noimp` 为：

- `epoch 5`: `55.06% / 87.07%`
- `epoch 10`: `64.43% / 91.53%`
- `epoch 15`: `74.00% / 95.11%`
- `epoch 20`: `73.77% / 95.04%`

两者差异很小。`A4` 的最终 `Top-1` 仅比历史 `spatial_noimp` 高 `+0.51`，而且曲线形态没有出现 `distance` 那种强烈震荡。

这说明：  
“先整体归一化再切分”带来的语义问题，主要集中在 `distance partitioning`，并没有在 `spatial partitioning` 上表现为同量级异常。

## 5. 结果可视化产物

本轮已经整理出两张专门图像：

- 实验结果总图：
  - [work_dir/figures/ntu_xsub_distance_compensation_results.png](../work_dir/figures/ntu_xsub_distance_compensation_results.png)
  - [work_dir/figures/ntu_xsub_distance_compensation_results.pdf](../work_dir/figures/ntu_xsub_distance_compensation_results.pdf)
- 权重分析图：
  - [work_dir/figures/ntu_xsub_distance_compensation_weights.png](../work_dir/figures/ntu_xsub_distance_compensation_weights.png)
  - [work_dir/figures/ntu_xsub_distance_compensation_weights.pdf](../work_dir/figures/ntu_xsub_distance_compensation_weights.pdf)

对应脚本为：

- [tools/plot_distance_compensation_results.py](../tools/plot_distance_compensation_results.py)
- [tools/export_effective_adjacency_stats.py](../tools/export_effective_adjacency_stats.py)

导出的 `A_eff` 统计文件位于：

- [distance_imp_e40/stats/effective_adjacency.csv](../work_dir/ablation/ntu-xsub/distance_imp_e40/stats/effective_adjacency.csv)
- [distance_subsetnorm_rescaled_imp_e40/stats/effective_adjacency.csv](../work_dir/ablation/ntu-xsub/distance_subsetnorm_rescaled_imp_e40/stats/effective_adjacency.csv)

## 6. 实验结果分析

### 6.1 `importance` 能显著补偿原始 `distance` 异常

这是本轮最直接的结论之一。

历史 `distance_noimp` 到 `epoch 20` 只有 `61.85%`，而当前：

- `A1 = distance_imp` 达到 `77.12%`
- `A3 = distance_subsetnorm_rescaled_imp` 达到 `77.59%`

也就是说，只要给 `distance` 打开 `edge importance weighting`，原本最异常的一组配置就能立刻回到接近正常结构模型的区间。

这说明：  
原始 `distance` 的问题并不只是“天生比 `uniform` 或 `spatial` 差”，而是它在无可学习边权时，确实对 subset balance 更敏感。

### 6.2 `subsetnorm_rescaled` 本身也有独立收益

如果 `importance` 已经完全解决了问题，那么 `A3/B2` 理应与 `A1/B1` 基本重合。

但实际不是这样：

- 阶段一 `20 epoch`，`A3` 在 `epoch 4 / 9 / 14 / 19` 全部略优于 `A1`
- 阶段二 `40 epoch`，`B2` 最终比 `B1` 高 `+1.69 Top-1`

也就是说，`importance` 确实在补偿原版 `distance` 的不理想平衡，但它不是全部。  
在“总消息量不被放大”的前提下，把 `distance` 改成 `subset-wise normalize + 1/K rescale`，仍然带来了一段可保留的净收益。

### 6.3 `subsetnorm_rescaled_noimp` 的价值主要在“稳住”，不是直接追平 `imp`

`A2 = distance_subsetnorm_rescaled_noimp` 的最终精度仍显著低于带 `importance` 的两组，但它的曲线已经明显脱离了原始 `distance_noimp` 那种“中期塌陷又回升”的形态。

更准确地说：

- 原始 `distance_noimp`：验证强烈震荡
- `A2`：验证曲线基本单调上升，到 `epoch 20` 稳定在 `72.57%`

所以 `A2` 的价值不在于直接证明“光改归一化就够了”，而在于证明：

- subset balance 偏置本身确实是问题来源之一
- 即便不借助 `importance`，只把归一化语义修平，也能先把最明显的异常形态压住

### 6.4 该问题主要集中在 `distance`，而不是普遍存在于所有 partition

`A4 = spatial_subsetnorm_rescaled_noimp` 与原始 `spatial_noimp` 之间只有小幅差异，没有出现类似 `distance` 的异常收益或异常修复。

这说明：

- `spatial partitioning` 的结构先验更强
- 即使原实现口径不是严格的 subset-wise normalize，它也没有在 `spatial` 上表现为同等级失衡
- 因而当前问题更适合被归类为 `distance partitioning` 的特定实现偏置，而不是整个 ST-GCN 多子集实现都需要重写

## 7. 权重分析

### 7.1 `mask_mean` 本身没有显示出强烈的不对称

对两组 40 epoch `importance` 模型，`self / neighbor` 的 `mask_mean` 比值均值分别为：

- `distance_imp_e40`: `1.017`
- `distance_subsetnorm_rescaled_imp_e40`: `1.009`

这说明如果只看 `M` 的平均值，很容易得出“两个子集权重差不多”的表面结论。

但这并不足以解释最终性能差异，因为 `M` 是乘在基础邻接 `A` 上的。

### 7.2 真正关键的是 `A_eff = A ⊙ M` 的子集强度分布

对两组模型的 `self / neighbor` `a_eff_colsum_mean` 比值均值进行比较，可得到：

- `distance_imp_e40`: `0.778`
- `distance_subsetnorm_rescaled_imp_e40`: `1.226`

进一步看每组模型的子集列和均值：

- `distance_imp_e40`
  - self mean: `0.2002`
  - neighbor mean: `0.2710`
- `distance_subsetnorm_rescaled_imp_e40`
  - self mean: `0.2844`
  - neighbor mean: `0.2370`

这说明：

1. 原版 `distance_imp` 虽然已经有 `importance`，但有效邻接整体上仍然是 **neighbor-dominant**。  
   尤其在更深层，self 分支强度持续偏弱，`layer 10` 的 self / neighbor 比值甚至只有 `0.374`。

2. `distance_subsetnorm_rescaled_imp` 则明显把有效邻接重心拉回到 **self-balanced / self-favoring**。  
   它在前中层多次保持 `self / neighbor > 1`，例如：
   - `layer 1`: `1.492`
   - `layer 3`: `1.666`
   - `layer 5`: `1.400`
   - `layer 7`: `1.255`

### 7.3 这与补偿假设是吻合的

如果原版 `distance` 的基础口径就让 self 分支偏弱，那么：

- 仅看 `M` 平均值，未必能看出明显补偿
- 但看 `A_eff = A ⊙ M`，就会发现模型最终仍然留下了“neighbor 主导”的有效图结构

而 `subsetnorm_rescaled` 的作用不是简单把 `M` 学大，而是先在基础邻接层面把 subset balance 做平，然后让 `importance` 在这个更公平的底座上继续学习。

因此当前最合理的解读是：

- `importance` 在原版 `distance` 上确实承担了部分补偿角色
- 但这种补偿并不彻底
- 一旦把底层归一化口径改成 `subsetnorm_rescaled`，模型就更容易形成对 self 分支更健康的有效邻接分布，并进一步转化为最终精度收益

## 8. 当前结论

基于这轮六组实验与两组 40 epoch 权重统计，目前可以写出较稳妥的结论：

1. `edge importance weighting` 能显著补偿原版 `distance` 的异常。  
   这是最直接、最明确的结论。

2. `subsetnorm_rescaled` 仍然存在独立于 `importance` 的净收益。  
   否则 `distance_subsetnorm_rescaled_imp` 不会在 20 epoch 和 40 epoch 两个阶段都稳定优于 `distance_imp`。

3. 原版 `distance` 的问题更像是 subset balance 偏置，而不是切分错误、support 不守恒或前向公式错误。  
   这点与前面的语义审计结果一致。

4. 该实现口径问题主要集中在 `distance partitioning`。  
   `spatial_subsetnorm_rescaled_noimp` 没有表现出同量级变化。

5. `A_eff` 分析支持“importance 在补偿原版偏置”这一假设链，但也表明补偿并不完全。  
   `subsetnorm_rescaled` 让最终有效邻接从 neighbor-dominant 转向更平衡甚至 self-favoring 的状态，这与最终性能提升方向一致。

## 9. 后续建议

如果后续继续沿这条线推进，我建议优先做两件事，而不是重新扩张实验面：

1. 将 `distance_subsetnorm_rescaled` 视为后续 `distance partitioning` 相关实验的更优实现口径。  
   当前证据已经足够说明它不是偶然改好一次曲线。

2. 若还想继续验证“该问题是否只集中在 `distance`”，可以选择性补一组 `spatial_subsetnorm_rescaled_imp`。  
   但这不是当前最紧急的工作，因为现有结果已经足够支持“影响主要集中在 `distance`”这一结论。
