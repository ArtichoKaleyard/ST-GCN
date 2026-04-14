#!/usr/bin/env python
"""ST-GCN 命令行入口。

保持官方仓库的子命令结构与参数解析行为不变，只将实现整理为更现代、
更易维护的 Python 写法。
"""

import argparse
import sys

import torchlight
from torchlight import import_class


def build_processor_registry() -> dict[str, type]:
    """构建可用处理器注册表。

    返回的 key 必须继续对齐官方仓库的子命令名。这样既能保持
    `main.py recognition ...` 这类入口兼容，也能让历史文档和旧脚本
    不需要因为 modern 分支的代码整理而改命令。
    """
    return {
        "recognition": import_class("processor.recognition.REC_Processor"),
        "demo_old": import_class("processor.demo_old.Demo"),
        "demo": import_class("processor.demo_realtime.DemoRealtime"),
        "demo_offline": import_class("processor.demo_offline.DemoOffline"),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="处理器集合")
    processors = build_processor_registry()

    subparsers = parser.add_subparsers(dest="processor")
    for name, processor in processors.items():
        subparsers.add_parser(name, parents=[processor.get_parser()])

    arg = parser.parse_args()
    processor_cls = processors[arg.processor]
    processor = processor_cls(sys.argv[2:])
    processor.start()
