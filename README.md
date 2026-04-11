## 说明

ST-GCN 官方实现后续已迁移到 [MMSkeleton](https://github.com/open-mmlab/mmskeleton)，
原始仓库更适合作为历史实现与论文补充材料参考。

当前这个 `modern` 分支的目标不是改模型设计，而是在**不改变公开行为、参数定义和旧权重兼容性**的前提下，把仓库重写到更现代的 Python / PyTorch 代码风格：

- 主目标环境：`Python 3.12.x`
- 主目标框架：`PyTorch 2.x`
- 保留原有 `main.py` 子命令、YAML 配置键名、checkpoint 读写格式
- 保留 ST-GCN / Two-Stream ST-GCN 的数学结构与训练流程
- 中文化 README、CLI 帮助文案、关键注释与 docstring

## 当前仓库定位

本仓库对应论文：

> **Spatial Temporal Graph Convolutional Networks for Skeleton-Based Action Recognition**, Sijie Yan, Yuanjun Xiong and Dahua Lin, AAAI 2018. [[Arxiv Preprint]](https://arxiv.org/abs/1801.07455)

如果你需要的是官方后续维护版本、更多模型或更完整工具链，建议直接迁移到 MMSkeleton。
如果你需要的是更易维护、但接口仍尽量对齐官方仓库的 ST-GCN 实现，则继续使用本分支即可。

## 现代化改动原则

- 不改已有命令入口：
  `recognition`、`demo_old`、`demo`、`demo_offline`
- 不改已有参数名、配置名与默认行为
- 不主动引入会改变训练语义的增强：
  例如 `DDP`、`AMP`、`torch.compile`
- 保持旧模型权重加载兼容，包括 `module.` 前缀清理与 `ignore_weights` 逻辑
- 仅把实现升级为更现代、可维护的 Python / PyTorch 写法

## 环境说明

推荐使用 `uv` 管理环境与依赖。

```bash
uv sync
```

项目元数据已迁移到仓库根的 `pyproject.toml`。

注意：

- 该分支主目标是 `Python 3.12.x + PyTorch 2.x`
- 代码设计上保留旧接口兼容，但不承诺继续支持所有旧环境组合
- 具体 CUDA 版本应以你的驱动和 PyTorch 官方 wheel 对应关系为准

## 旧版说明

如果你想查看官方仓库原始说明与旧命令示例，请参考 [OLD_README.md](./OLD_README.md)。
