# ST-GCN 项目约束

## 项目定位

这是 ST-GCN 官方仓库的 `modern` 分支。

本分支的目标不是重设模型，而是在尽量不改变外部行为的前提下，把官方旧实现整理为更现代、可维护的 Python / PyTorch 代码：

- 保持 `main.py` 子命令结构不变
- 保持配置文件键名与参数名不变
- 保持旧 checkpoint 加载语义不变
- 保持 ST-GCN / Two-Stream ST-GCN 的核心数学结构与论文定义一致

只要用户没有明确要求，不要主动引入会改变训练数值路径或外部接口的增强项，例如：

- `torch.compile`
- AMP / autocast 默认开启
- DDP 替换现有 `DataParallel`
- 新的学习率调度器
- 改写配置格式或 CLI 参数命名

## 技术栈

- Python `3.12.x`
- PyTorch `2.x`
- 包管理使用 `uv`
- 入口仍以 `argparse + YAML config` 为主

## 关键命令

- 安装依赖：`uv sync`
- 查看总帮助：`.venv/bin/python main.py -h`
- 查看识别子命令帮助：`.venv/bin/python main.py recognition -h`
- 语法检查：`python3 -m compileall main.py processor net feeder tools torchlight`
- Bash 脚本语法检查：`bash -n <script>`

如果 `.venv` 已存在，优先直接使用 `.venv/bin/python` 做验证，不要依赖系统 `python3` 对项目状态下结论。

## 重要路径

- [main.py](/home/atk/PyCharm/ST-GCN/main.py): CLI 入口与处理器注册
- [processor/](/home/atk/PyCharm/ST-GCN/processor): 训练、测试和 demo 处理器
- [net/](/home/atk/PyCharm/ST-GCN/net): ST-GCN 与双流模型定义
- [feeder/](/home/atk/PyCharm/ST-GCN/feeder): 数据集与数据增强
- [config/](/home/atk/PyCharm/ST-GCN/config): 官方配置样例
- [torchlight/](/home/atk/PyCharm/ST-GCN/torchlight): 旧版辅助框架兼容层

## 工作流约束

- 修改模型或处理器时，先检查是否会改变以下外部契约：
  - CLI 参数名
  - YAML 配置键名
  - `state_dict` 键名与加载逻辑
  - 输入输出张量形状
- `net/st_gcn.py` 和 `net/st_gcn_twostream.py` 的变更必须优先按“与官方实现等价”审视，再考虑现代化风格。
- 双流模型允许修复“写死 CUDA tensor”这类设备兼容问题，但不能改 motion stream 的数学定义。
- 项目文档、模块说明、注释和 docstring 默认使用简体中文；公开 Python 符号名、配置键名、命令行参数名保持英文兼容。

## 已知风险

- `demo_old`、`demo`、`demo_offline` 依赖 OpenPose 或外部视频资源；没有真实 OpenPose 环境时，只能做导入级与参数级验证。
- 官方仓库历史上依赖单独安装 `torchlight/` 子包；当前分支通过仓库根 [torchlight/__init__.py](/home/atk/PyCharm/ST-GCN/torchlight/__init__.py) 兼容“直接在仓库根运行”和“作为包安装”两种入口。不要删除这层兼容。
- 若需要判断现代化重写是否仍对齐官方定义，优先直接对比 `git show HEAD:<path>` 的官方版本，而不是只凭当前代码风格判断。
