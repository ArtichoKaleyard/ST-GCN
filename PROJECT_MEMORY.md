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
