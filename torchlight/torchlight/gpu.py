"""GPU 相关辅助函数。"""

from __future__ import annotations

import os

import torch


def visible_gpu(gpus: int | list[int]) -> list[int]:
    """设置可见 GPU，并返回重映射后的逻辑设备编号列表。"""
    gpu_list = [gpus] if isinstance(gpus, int) else list(gpus)
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_list))
    return list(range(len(gpu_list)))


def ngpu(gpus: int | list[int]) -> int:
    """统计使用的 GPU 数量。"""
    gpu_list = [gpus] if isinstance(gpus, int) else list(gpus)
    return len(gpu_list)


def occupy_gpu(gpus: int | list[int] | None = None) -> None:
    """让程序在 `nvidia-smi` 中显式出现。"""
    if gpus is None:
        torch.zeros(1, device="cuda")
        return

    gpu_list = [gpus] if isinstance(gpus, int) else list(gpus)
    for gpu_id in gpu_list:
        torch.zeros(1, device=f"cuda:{gpu_id}")
