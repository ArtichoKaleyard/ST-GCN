#!/usr/bin/env python
"""训练/测试处理器基类。"""

from __future__ import annotations

import argparse
import random
from typing import Any

import numpy as np
import torch

import torchlight
from torchlight import DictAction
from torchlight import import_class
from torchlight import str2bool

from .io import IO


class Processor(IO):
    """训练/测试处理器基类。

    该类保留官方版本的阶段组织方式：`train()` / `test()` 负责单轮逻辑，
    `start()` 负责 epoch 级调度、模型保存与评估触发。
    """

    def __init__(self, argv: list[str] | None = None) -> None:
        self.load_arg(argv)
        self.init_environment()
        self.load_model()
        self.load_weights()
        self.gpu()
        self.load_data()
        self.load_optimizer()

    def init_environment(self) -> None:
        """初始化处理器上下文。"""
        super().init_environment()
        self.set_random_seed()
        self.result: dict[str, Any] | np.ndarray = {}
        self.iter_info: dict[str, Any] = {}
        self.epoch_info: dict[str, Any] = {}
        self.meta_info: dict[str, int] = dict(epoch=0, iter=0)

    def set_random_seed(self) -> None:
        """在主进程与 CUDA 上统一设置随机种子。

        这里的目标不是追求 PyTorch 所有算子的绝对逐 bit 可复现，而是把
        Python / NumPy / Torch / DataLoader worker / shuffle 的随机源统一到
        同一个显式 seed 上，避免“默认单次运行”被误写成“统一 seed 实验”。
        """
        seed = int(self.arg.seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # 保持 cudnn benchmark 与项目旧训练路径一致，不强行切到完全确定性
        # 模式；这轮需求是“显式固定同一 seed”，不是额外改动数值/性能路径。
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True

        # 为 DataLoader 的 shuffle 与 worker 初始化保留同一个基础随机源。
        self.data_loader_generator = torch.Generator()
        self.data_loader_generator.manual_seed(seed)

    def seed_worker(self, worker_id: int) -> None:
        """按 worker 派生随机种子，保持多进程数据加载可追踪。"""
        worker_seed = (int(self.arg.seed) + worker_id) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    def load_optimizer(self) -> None:
        """由子类实现优化器加载。"""

    def load_data(self) -> None:
        """构建训练/测试数据加载器。"""
        feeder_cls = import_class(self.arg.feeder)

        # 旧实现只把 `--debug` 自动注入训练 feeder；测试 feeder 是否裁剪数据
        # 继续由调用侧显式决定，避免悄悄改评估集大小。
        if "debug" not in self.arg.train_feeder_args:
            self.arg.train_feeder_args["debug"] = self.arg.debug

        self.data_loader: dict[str, torch.utils.data.DataLoader] = {}
        if self.arg.phase == "train":
            self.data_loader["train"] = torch.utils.data.DataLoader(
                dataset=feeder_cls(**self.arg.train_feeder_args),
                batch_size=self.arg.batch_size,
                shuffle=True,
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                drop_last=True,
                worker_init_fn=self.seed_worker,
                generator=self.data_loader_generator,
            )
        if self.arg.test_feeder_args:
            self.data_loader["test"] = torch.utils.data.DataLoader(
                dataset=feeder_cls(**self.arg.test_feeder_args),
                batch_size=self.arg.test_batch_size,
                shuffle=False,
                num_workers=self.arg.num_worker * torchlight.ngpu(self.arg.device),
                worker_init_fn=self.seed_worker,
                generator=self.data_loader_generator,
            )

    def show_epoch_info(self) -> None:
        """打印轮级别统计信息。"""
        for key, value in self.epoch_info.items():
            self.io.info(f"\t{key}: {value}")
        if self.arg.pavi_log:
            self.io.log("train", self.meta_info["iter"], self.epoch_info)

    @staticmethod
    def _format_metric_value(value: Any) -> str:
        """将日志指标格式化为紧凑字符串。"""
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    def get_iter_postfix(self) -> dict[str, str]:
        """返回适合 tqdm postfix 的迭代指标。"""
        return {
            key: self._format_metric_value(value)
            for key, value in self.iter_info.items()
        }

    def show_iter_info(self) -> None:
        """按设定间隔记录迭代信息。"""
        if self.meta_info["iter"] % self.arg.log_interval == 0:
            info = f"\tIter {self.meta_info['iter']} Done."
            for key, value in self.iter_info.items():
                info += f" | {key}: {self._format_metric_value(value)}"

            self.io.debug(info)

            if self.arg.pavi_log:
                self.io.log("train", self.meta_info["iter"], self.iter_info)

    def train(self) -> None:
        """占位训练逻辑。"""
        for _ in range(100):
            self.iter_info["loss"] = 0
            self.show_iter_info()
            self.meta_info["iter"] += 1
        self.epoch_info["mean loss"] = 0
        self.show_epoch_info()

    def test(self) -> None:
        """占位测试逻辑。"""
        for _ in range(100):
            self.iter_info["loss"] = 1
            self.show_iter_info()
        self.epoch_info["mean loss"] = 1
        self.show_epoch_info()

    def start(self) -> None:
        """启动训练或评测流程。"""
        self.io.info("Parameters:\n{}\n".format(str(vars(self.arg))))

        if self.arg.phase == "train":
            for epoch in range(self.arg.start_epoch, self.arg.num_epoch):
                self.meta_info["epoch"] = epoch

                # 训练、存档、评估三段顺序保持与官方处理器一致。
                self.io.info(f"Training epoch: {epoch}")
                self.train()
                self.io.success(f"Training epoch {epoch} done.")

                if ((epoch + 1) % self.arg.save_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch
                ):
                    filename = f"epoch{epoch + 1}_model.pt"
                    self.io.save_model(self.model, filename)

                if ((epoch + 1) % self.arg.eval_interval == 0) or (
                    epoch + 1 == self.arg.num_epoch
                ):
                    self.io.info(f"Eval epoch: {epoch}")
                    self.test()
                    self.io.success(f"Eval epoch {epoch} done.")
        elif self.arg.phase == "test":
            if self.arg.weights is None:
                raise ValueError("Please appoint --weights.")
            self.io.info(f"Model:   {self.arg.model}.")
            self.io.info(f"Weights: {self.arg.weights}.")

            self.io.info("Evaluation Start:")
            self.test()
            self.io.success("Evaluation done.\n")

            if self.arg.save_result:
                result_dict = dict(
                    zip(self.data_loader["test"].dataset.sample_name, self.result)
                )
                self.io.save_pkl(result_dict, "test_result.pkl")

    @staticmethod
    def get_parser(add_help: bool = False) -> argparse.ArgumentParser:
        """获取处理器通用参数解析器。"""
        parser = argparse.ArgumentParser(add_help=add_help, description="Base Processor")

        parser.add_argument(
            "-w",
            "--work_dir",
            default="./work_dir/tmp",
            help="保存结果的工作目录",
        )
        parser.add_argument("-c", "--config", default=None, help="配置文件路径")

        parser.add_argument("--phase", default="train", help="必须是 train 或 test")
        parser.add_argument(
            "--save_result",
            type=str2bool,
            default=False,
            help="若为真，则保存模型输出结果",
        )
        parser.add_argument(
            "--start_epoch", type=int, default=0, help="从哪个 epoch 开始训练"
        )
        parser.add_argument(
            "--num_epoch", type=int, default=80, help="训练到哪个 epoch 结束"
        )
        parser.add_argument("--use_gpu", type=str2bool, default=True, help="是否使用 GPU")
        parser.add_argument(
            "--device",
            type=int,
            default=0,
            nargs="+",
            help="训练或测试所用 GPU 编号",
        )

        parser.add_argument(
            "--log_interval", type=int, default=100, help="日志打印间隔（迭代数）"
        )
        parser.add_argument(
            "--save_interval", type=int, default=10, help="模型保存间隔（epoch 数）"
        )
        parser.add_argument(
            "--eval_interval", type=int, default=5, help="评估间隔（epoch 数）"
        )
        parser.add_argument(
            "--save_log", type=str2bool, default=True, help="是否保存日志"
        )
        parser.add_argument(
            "--print_log", type=str2bool, default=True, help="是否打印日志"
        )
        parser.add_argument("--pavi_log", type=str2bool, default=False, help="是否启用 pavi 日志")

        parser.add_argument("--feeder", default="feeder.feeder", help="使用的数据加载器")
        parser.add_argument(
            "--num_worker", type=int, default=4, help="每个 GPU 的 data loader worker 数"
        )
        parser.add_argument(
            "--train_feeder_args",
            action=DictAction,
            default=dict(),
            help="训练数据加载器参数",
        )
        parser.add_argument(
            "--test_feeder_args",
            action=DictAction,
            default=dict(),
            help="测试数据加载器参数",
        )
        parser.add_argument("--batch_size", type=int, default=256, help="训练 batch size")
        parser.add_argument(
            "--test_batch_size", type=int, default=256, help="测试 batch size"
        )
        parser.add_argument("--debug", action="store_true", help="少量数据，便于快速调试")
        parser.add_argument(
            "--seed",
            type=int,
            default=49,
            help="统一实验随机种子，用于主进程、CUDA 与 DataLoader worker",
        )

        parser.add_argument("--model", default=None, help="使用的模型")
        parser.add_argument("--model_args", action=DictAction, default=dict(), help="模型参数")
        parser.add_argument("--weights", default=None, help="网络初始化权重路径")
        parser.add_argument(
            "--ignore_weights",
            type=str,
            default=[],
            nargs="+",
            help="初始化时忽略的权重名前缀",
        )

        return parser
