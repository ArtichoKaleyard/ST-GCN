# PROJECT_MEMORY

## Modern 分支迁移结论

- 这个仓库历史上对 `torchlight` 的使用依赖“单独安装 `torchlight/` 子包”；在 modern 分支引入仓库根 `pyproject.toml` 后，若不补仓库根 [torchlight/__init__.py](/home/atk/PyCharm/ST-GCN/torchlight/__init__.py)，直接运行 `python main.py` 只会拿到 namespace package，`import torchlight` 不具备 `IO` / `import_class` 等接口。
- 当前 modern 分支已通过仓库根兼容层恢复两种入口：
  1. 直接在仓库根运行 `main.py`
  2. 作为包安装后导入 `torchlight`
- Python 版本约束不应写成宽泛的 `>=3.12`，否则会把未来未验证的 `3.14+` 自动纳入兼容面；当前项目已收敛为 `>=3.12,<3.13`。
- 在本节点上，`uv sync` 已成功创建 `.venv`，并实际使用 `CPython 3.12.12`。后续做 modern 分支验证时，优先直接使用 `.venv/bin/python`，不要继续用系统 `python3` 推断仓库状态。
- `uv run` 在当前沙箱里可能因 `~/.cache/uv` 不可写而失败；若 `.venv` 已经存在，优先直接调用 `.venv/bin/python` 做验证，避免把缓存权限问题误判成项目问题。
- 已做过的有效验证：
  - `.venv/bin/python main.py -h`
  - `.venv/bin/python main.py recognition -h`
  - 单流 `net.st_gcn.Model` 前向与 `extract_feature` 输出形状验证
  - 双流 `net.st_gcn_twostream.Model` 前向形状验证
- 与 `git show HEAD:net/st_gcn.py` / `git show HEAD:net/utils/graph.py` 的对照验证已经做过：
  - 图邻接矩阵完全一致
  - `eval()` 下单流前向输出完全一致
  - `eval()` 下 `extract_feature` 的输出与特征张量完全一致
  - 双流模型的实质性变化仅是把原版写死的 `torch.cuda.FloatTensor(...)` 改成了 device-agnostic 写法，数学公式未改

## 当前已知缺陷 / 未完成验证

- `demo_old`、`demo`、`demo_offline` 还没有在真实 OpenPose 环境下做端到端运行验证；当前只有代码级迁移、参数解析和导入级保证。
- `demo_offline` 在最初的 modern 重写里曾误引入两处与官方不一致的 tracker 行为：`get_skeleton_sequence()` 对 trace 按分数排序并截断到前 2 人，以及 `get_dis()` 使用了对角线尺度 `sqrt(w^2 + h^2)`。这两处现已回滚到官方语义；后续若再动 demo 逻辑，必须先明确这是兼容修复还是有意行为变更。
- `DemoOffline.pose_estimation()` 当前已改为在 OpenPose Python API 缺失时显式返回 `None`，`start()` 会抛出清晰异常而不是沿用官方版的隐式解包失败；这属于失败路径诊断改进，不影响成功路径。
- `Graph.__str__` 已从返回 `np.ndarray` 对象改为返回字符串，属于纯代码卫生修复，不影响图构造与模型计算。
- 与官方实现的严格数值等价性，目前只验证到了 `eval()` 路径；`train()` 态下出现数值不完全一致主要来自 dropout / BatchNorm 的训练时随机性与状态更新，不应直接解读为结构偏离，但也还没有做更细的逐层训练态对照。
- 当前还没有用官方预训练权重做一次完整的 `recognition --phase test` 回归，因此旧 checkpoint 的加载语义虽然保留，仍缺少一次真实数据集上的端到端结果确认。

## 2026-04-12 - Herald 与 tqdm 日志改造
- Context: 用户要求把训练时的 iter 刷屏改成 `tqdm` 进度条，并接入 `Herald`，终端隐藏 debug 级 iter 明细，但文件日志尽量保留原有完整输出。
- Decision: 在 `torchlight.IO` 内集中接入 `Herald`，使用 `ConsoleHandler(level="info") + FileHandler(level="debug")`；训练和测试循环单独使用 `tqdm` 展示 iter 进度，不让 iter 文本再走终端日志。
- Why: `Herald` 的 `Logger` 先做全局级别过滤，再由 Handler 做二次过滤；因此若要保留文件中的 debug iter 明细，logger 本身必须保持 `debug`，只把终端 Handler 收紧到 `info`。`tqdm` 不能视为普通 info 日志，否则会被级别策略误伤。
- Action/Command: `uv add tqdm 'Herald @ git+https://github.com/ArtichoKaleyard/Herald'`；`.venv/bin/python -m compileall main.py processor net feeder tools torchlight`；`.venv/bin/python main.py recognition -h`；用 `torchlight.IO('temp/herald_probe', save_log=True, print_log=True)` 做最小日志探针，确认控制台隐藏 debug、`log.txt` 保留 debug。
- Verification: 已确认 `Herald` 可正常导入；控制台只显示 `info/success`，文件日志包含 `debug`；CLI 帮助与语法检查通过。
- Follow-up: 后续若继续收敛噪声，可再评估是否把逐层权重加载明细长期保留在 `debug`，并用真实训练配置检查 `tqdm` 在长跑任务中的观感。

## 2026-04-12 - 两次正式训练结果可视化
- Context: 用户要求把同一数据集上的两轮正式训练结果做成学术排版对比图；当前目标是对比仓库默认配置与论文学习率日程配置，不手工计算子图坐标。
- Decision: 新增 [tools/plot_training_comparison.py](/home/atk/PyCharm/ST-GCN/tools/plot_training_comparison.py)，默认直接读取 `work_dir/recognition/ntu-xsub/ST_GCN_fp32_bs16_lr0025_e80` 与 `work_dir/recognition/ntu-xsub/ST_GCN_fp32_bs16_lr0025_e80_step10x`，统一抽取训练均值 loss、验证 Top-1/Top-5 与 epoch 级学习率，并用 `subplot_mosaic(..., constrained_layout=True)` 输出 2x2 对比图。
- Why: 当前仓库内两次正式训练的 `log.txt` 并非同一种格式。默认配置目录仍是旧版时间戳文本，论文配置目录则是 Herald 的多列日志；若只按一种格式写解析器，后续图脚本会立即失效。稳定锚点是 `Training epoch`、`mean_loss`、`Eval epoch`、`Top1/Top5` 与 iter 行中的 `lr`。
- Action/Command: 使用 `.venv/bin/python tools/plot_training_comparison.py` 实际渲染；首次运行因缺少 `matplotlib` 失败后，按项目约束执行 `uv add matplotlib` 补齐依赖；为避免 Matplotlib 在当前环境里因 `~/.config/matplotlib` 不可写而报缓存目录警告，脚本入口显式设置 `MPLCONFIGDIR=ROOT_DIR/temp/matplotlib`。
- Verification: `python3 -m compileall tools/plot_training_comparison.py` 通过；已成功导出 [ntu_xsub_training_comparison.png](/home/atk/PyCharm/ST-GCN/work_dir/figures/ntu_xsub_training_comparison.png) 与 [ntu_xsub_training_comparison.pdf](/home/atk/PyCharm/ST-GCN/work_dir/figures/ntu_xsub_training_comparison.pdf)。当前两次正式训练的最终验证指标为：默认配置 `Top1=78.75% / Top5=96.32%`，论文学习率日程 `Top1=79.90% / Top5=96.73%`。
- Follow-up: 这份脚本目前默认绑定 NTU xsub 的两组正式训练目录；如果后续要支持更多实验组合，优先扩展 `--run LABEL WORK_DIR` 输入，而不是在脚本内部继续写死新的目录分支。

## 2026-04-13 - 第一轮消融实验协议
- Context: 用户准备先做 NTU RGB+D xsub 的第一轮消融，只跑当前仓库已有实现底座的 4 项：`uniform_noimp`、`distance_noimp`、`spatial_noimp`、`spatial_imp`；目标是先看相对趋势，不追求一步到位复现论文最终绝对数值。
- Decision: 第一轮统一协议固定为 `40 epoch + eval_interval=5 + 单 seed + 单卡 batch_size=16 + base_lr=0.025 + step=[10,20,30] + AMP`，并把 4 份配置放到 `config/ablation/ntu-xsub/`，对应工作目录放到 `work_dir/ablation/ntu-xsub/`。
- Why: 当前正式训练曲线已确认 40 轮后收益很弱，继续拉长到 80 轮会显著增加时间成本；但直接缩到 30 轮又容易把“谁前期起步快”误当成结构结论。对 40 轮短程协议而言，把长程多段衰减显式裁成有效前缀 `step=[10,20,30]` 不改变可达 epoch 内的行为，只是去掉不可达的后续衰减点。AMP 在这轮实验里视为统一训练协议的一部分，而不是单独消融变量。
- Action/Command: 新增 4 份 ablation config，并在 `processor/recognition.py` 补充 `--amp` 开关，使用 `torch.amp.autocast` 与 `GradScaler` 支持 CUDA AMP 训练/评测。
- Verification: 后续至少应做 `recognition -h` 参数检查，以及逐份配置的 YAML 读取检查和基础语法检查；真正的趋势判断以 4 项实验全部跑完后的 Top-1 排序为准。
- Follow-up: 第一轮完成后，优先检查 `spatial_imp > spatial_noimp` 以及 `spatial` 系列相对 `uniform/distance` 的排序是否合理；`baseline_tcn` 与 `local_conv` 留到第二轮，不在本次配置里提前混入。

## 2026-04-13 - 阶段二消融预研
- Context: 用户要求先只跑第一轮 4 个现成消融，但希望提前预研第二轮大致该怎么做。
- Decision: 阶段二优先顺序定为 `baseline_tcn` -> `local_conv`。`baseline_tcn` 适合做成一个最小纯时域卷积模型，单独回答“去掉图结构会掉多少”；`local_conv` 暂不实现，只保留技术设计预研，因为它最容易在“局部连接不共享权重”的定义上做偏。
- Why: 当前仓库的 ST-GCN 主干和 `net/utils/tgcn.py` 已经把“共享卷积 + 邻接矩阵聚合”的结构拆清楚了，做 `baseline_tcn` 风险低、解释也直接；而 `local_conv` 的风险不在代码量，而在定义是否仍符合论文语义，以及参数量是否会失控。
- Action/Command: 预研结论是后续若进入阶段二，优先新增 `net/baseline_tcn.py` 与对应 ablation config；`local_conv` 则优先新增 `net/utils/local_tgcn.py` 设计稿，再决定是否补 `net/local_conv.py`，不要直接边写边改主模型。
- Verification: 当前只是方案预研，不涉及代码实现或数值验证。
- Follow-up: 真正进入阶段二前，先锁死 `local_conv` 的定义边界：是否沿用 `spatial` partition、参数是否按 node 独立、是否需要约束参数量，避免写成“另一个新模型”而不是论文消融项。

## 2026-04-13 - AMP 最短探针
- Context: 在正式串行跑 4 个第一轮消融之前，用户要求先做一个最短临时测试，对比 AMP 是否会在当前训练链路里引入明显异常。
- Decision: 基于 `config/ablation/ntu-xsub/spatial_imp.yaml`，各跑一组 `debug + 1 epoch + eval_interval=1` 的临时探针，工作目录分别为 `temp/amp_probe_off` 与 `temp/amp_probe_on`。
- Why: 用户只要求验证 AMP 是否“可放心作为统一实验协议的一部分”，而不是要做严格 AMP 消融；因此最小可比探针就够了，不需要额外延长训练。
- Action/Command: `.venv/bin/python main.py recognition -c config/ablation/ntu-xsub/spatial_imp.yaml --work_dir temp/amp_probe_off --num_epoch 1 --save_interval 1 --eval_interval 1 --debug --amp False`；对应 AMP 开启版只把 `--amp True` 和输出目录改为 `temp/amp_probe_on`。
- Verification: 关闭 AMP 时得到 `Top1=1.67% / Top5=8.31% / mean_loss=4.09985`；开启 AMP 时得到 `Top1=1.67% / Top5=8.33% / mean_loss=4.09908`。两者在这组最短探针上基本一致，未观察到明显异常。
- Follow-up: 后续正式 40 epoch 消融默认可以继续使用 AMP；若最终排序出现可疑翻转，再回头把 AMP 单独拿出来做更长程对照。
