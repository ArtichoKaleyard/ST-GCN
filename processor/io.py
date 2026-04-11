#!/usr/bin/env python
"""处理器 I/O 基类。"""

from __future__ import annotations

import argparse
from typing import Any

import torch
import torch.nn as nn
import yaml

import torchlight
from torchlight import DictAction
from torchlight import str2bool


class IO:
    """IO Processor。"""

    def __init__(self, argv: list[str] | None = None):
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()

    def load_arg(self, argv: list[str] | None = None) -> None:
        """读取命令行参数与配置文件。"""
        parser = self.get_parser()
        parsed_args = parser.parse_args(argv)
        if parsed_args.config is not None:
            with open(parsed_args.config, "r", encoding="utf-8") as file_obj:
                default_arg = yaml.load(file_obj, Loader=yaml.FullLoader)

            valid_keys = vars(parsed_args).keys()
            for key in default_arg.keys():
                if key not in valid_keys:
                    print(f"Unknown Arguments: {key}")
                    assert key in valid_keys

            parser.set_defaults(**default_arg)

        self.arg = parser.parse_args(argv)

    def init_environment(self) -> None:
        """初始化日志、工作目录与设备。"""
        self.io = torchlight.IO(
            self.arg.work_dir,
            save_log=self.arg.save_log,
            print_log=self.arg.print_log,
        )
        self.io.save_arg(self.arg)

        if self.arg.use_gpu:
            gpus = torchlight.visible_gpu(self.arg.device)
            torchlight.occupy_gpu(gpus)
            self.gpus = gpus
            self.dev = torch.device("cuda:0")
        else:
            self.gpus = []
            self.dev = torch.device("cpu")

    def load_model(self) -> None:
        """实例化模型。"""
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args)

    def load_weights(self) -> None:
        """按需加载预训练权重。"""
        if self.arg.weights:
            self.model = self.io.load_weights(
                self.model, self.arg.weights, self.arg.ignore_weights
            )

    def gpu(self) -> None:
        """将模型和已挂载模块迁移到目标设备。"""
        self.model = self.model.to(self.dev)
        for name, value in vars(self).items():
            if isinstance(value, nn.Module):
                setattr(self, name, value.to(self.dev))

        if self.arg.use_gpu and len(self.gpus) > 1:
            self.model = nn.DataParallel(self.model, device_ids=self.gpus)

    def start(self) -> None:
        """打印参数。"""
        self.io.print_log("Parameters:\n{}\n".format(str(vars(self.arg))))

    @staticmethod
    def get_parser(add_help: bool = False) -> argparse.ArgumentParser:
        """获取 IO 基础参数解析器。"""
        parser = argparse.ArgumentParser(add_help=add_help, description="IO Processor")

        parser.add_argument(
            "-w",
            "--work_dir",
            default="./work_dir/tmp",
            help="保存结果的工作目录",
        )
        parser.add_argument(
            "-c", "--config", default=None, help="配置文件路径"
        )

        parser.add_argument(
            "--use_gpu", type=str2bool, default=True, help="是否使用 GPU"
        )
        parser.add_argument(
            "--device",
            type=int,
            default=0,
            nargs="+",
            help="训练或测试所用 GPU 编号",
        )

        parser.add_argument(
            "--print_log", type=str2bool, default=True, help="是否打印日志"
        )
        parser.add_argument(
            "--save_log", type=str2bool, default=True, help="是否保存日志"
        )

        parser.add_argument("--model", default=None, help="要使用的模型")
        parser.add_argument(
            "--model_args", action=DictAction, default=dict(), help="模型参数"
        )
        parser.add_argument(
            "--weights", default=None, help="网络初始化权重路径"
        )
        parser.add_argument(
            "--ignore_weights",
            type=str,
            default=[],
            nargs="+",
            help="初始化时忽略的权重名前缀",
        )

        return parser
