#!/usr/bin/env python
"""骨架动作识别处理器。"""

from __future__ import annotations

import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm

from torchlight import str2bool

from .processor import Processor


def weights_init(module: nn.Module) -> None:
    """保持官方初始化策略不变。"""
    class_name = module.__class__.__name__
    with torch.no_grad():
        if class_name.find("Conv1d") != -1:
            module.weight.normal_(0.0, 0.02)
            if module.bias is not None:
                module.bias.fill_(0)
        elif class_name.find("Conv2d") != -1:
            module.weight.normal_(0.0, 0.02)
            if module.bias is not None:
                module.bias.fill_(0)
        elif class_name.find("BatchNorm") != -1:
            module.weight.normal_(1.0, 0.02)
            module.bias.fill_(0)


class REC_Processor(Processor):
    """Processor for Skeleton-based Action Recognition。"""

    def load_model(self) -> None:
        """加载模型并初始化损失函数。"""
        self.model = self.io.load_model(self.arg.model, **self.arg.model_args)
        self.model.apply(weights_init)
        self.loss = nn.CrossEntropyLoss()

    def load_optimizer(self) -> None:
        """按参数选择优化器。"""
        if self.arg.optimizer == "SGD":
            self.optimizer = optim.SGD(
                self.model.parameters(),
                lr=self.arg.base_lr,
                momentum=0.9,
                nesterov=self.arg.nesterov,
                weight_decay=self.arg.weight_decay,
            )
        elif self.arg.optimizer == "Adam":
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=self.arg.base_lr,
                weight_decay=self.arg.weight_decay,
            )
        else:
            raise ValueError()

    def adjust_lr(self) -> None:
        """保持原版 SGD 阶梯式学习率衰减。"""
        if self.arg.optimizer == "SGD" and self.arg.step:
            lr = self.arg.base_lr * (
                0.1 ** np.sum(self.meta_info["epoch"] >= np.array(self.arg.step))
            )
            for param_group in self.optimizer.param_groups:
                param_group["lr"] = lr
            self.lr = lr
        else:
            self.lr = self.arg.base_lr

    def show_topk(self, k: int) -> None:
        """打印 top-k 准确率。"""
        rank = self.result.argsort()
        hit_top_k = [label in rank[i, -k:] for i, label in enumerate(self.label)]
        accuracy = sum(hit_top_k) * 1.0 / len(hit_top_k)
        self.io.info("\tTop{}: {:.2f}%".format(k, 100 * accuracy))

    def train(self) -> None:
        """训练一个 epoch。"""
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader["train"]
        loss_value: list[float] = []

        progress = tqdm(
            loader,
            desc=f"Train {self.meta_info['epoch']}",
            leave=False,
            dynamic_ncols=True,
        )
        for data, label in progress:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            output = self.model(data)
            loss = self.loss(output, label)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            self.iter_info["loss"] = loss.item()
            self.iter_info["lr"] = f"{self.lr:.6f}"
            loss_value.append(self.iter_info["loss"])
            progress.set_postfix(self.get_iter_postfix(), refresh=False)
            self.show_iter_info()
            self.meta_info["iter"] += 1

        self.epoch_info["mean_loss"] = np.mean(loss_value)
        self.show_epoch_info()
        self.io.print_timer()

    def test(self, evaluation: bool = True) -> None:
        """执行测试或纯推理。"""
        self.model.eval()
        loader = self.data_loader["test"]
        loss_value: list[float] = []
        result_frag: list[np.ndarray] = []
        label_frag: list[np.ndarray] = []

        progress = tqdm(
            loader,
            desc=f"Test {self.meta_info['epoch']}",
            leave=False,
            dynamic_ncols=True,
        )
        for data, label in progress:
            data = data.float().to(self.dev)
            label = label.long().to(self.dev)

            with torch.no_grad():
                output = self.model(data)
            result_frag.append(output.cpu().numpy())

            if evaluation:
                loss = self.loss(output, label)
                loss_value.append(loss.item())
                label_frag.append(label.cpu().numpy())
                progress.set_postfix({"loss": f"{loss.item():.4f}"}, refresh=False)

        self.result = np.concatenate(result_frag)
        if evaluation:
            self.label = np.concatenate(label_frag)
            self.epoch_info["mean_loss"] = np.mean(loss_value)
            self.show_epoch_info()

            for top_k in self.arg.show_topk:
                self.show_topk(top_k)

    @staticmethod
    def get_parser(add_help: bool = False) -> argparse.ArgumentParser:
        """获取识别任务参数解析器。"""
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description="Spatial Temporal Graph Convolution Network",
        )

        parser.add_argument(
            "--show_topk",
            type=int,
            default=[1, 5],
            nargs="+",
            help="显示哪些 Top-K 准确率",
        )
        parser.add_argument("--base_lr", type=float, default=0.01, help="初始学习率")
        parser.add_argument(
            "--step",
            type=int,
            default=[],
            nargs="+",
            help="学习率衰减发生的 epoch",
        )
        parser.add_argument("--optimizer", default="SGD", help="优化器类型")
        parser.add_argument(
            "--nesterov", type=str2bool, default=True, help="是否使用 nesterov"
        )
        parser.add_argument(
            "--weight_decay", type=float, default=0.0001, help="优化器权重衰减"
        )

        return parser
