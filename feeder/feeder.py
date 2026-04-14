"""通用骨架识别数据集。"""

from __future__ import annotations

import pickle

import numpy as np
import torch

from . import tools


class Feeder(torch.utils.data.Dataset):
    """通用骨架动作识别数据集。

    Args:
        data_path: 形如 ``(N, C, T, V, M)`` 的 `.npy` 数据文件路径。
        label_path: 与样本一一对应的标签文件路径。
        random_choose: 是否随机裁剪时间窗。
        random_move: 是否施加连续随机仿射扰动。
        window_size: 输出序列长度；小于等于 0 时保持原长度。
        debug: 是否只取前 100 个样本做快速调试。
        mmap: 是否以内存映射方式读取大数组。
    """

    def __init__(
        self,
        data_path: str,
        label_path: str,
        random_choose: bool = False,
        random_move: bool = False,
        window_size: int = -1,
        debug: bool = False,
        mmap: bool = True,
    ) -> None:
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_move = random_move
        self.window_size = window_size

        self.load_data(mmap)

    def load_data(self, mmap: bool) -> None:
        """加载标签与骨架数组。"""
        with open(self.label_path, "rb") as file_obj:
            self.sample_name, self.label = pickle.load(file_obj)

        if mmap:
            self.data = np.load(self.data_path, mmap_mode="r")
        else:
            self.data = np.load(self.data_path)

        if self.debug:
            self.label = self.label[0:100]
            self.data = self.data[0:100]
            self.sample_name = self.sample_name[0:100]

        self.N, self.C, self.T, self.V, self.M = self.data.shape

    def __len__(self) -> int:
        return len(self.label)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """返回单个样本与标签。"""
        data_numpy = np.array(self.data[index])
        label = self.label[index]

        # 增强顺序保持官方实现：先裁剪/补齐时间窗，再做随机运动扰动。
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        return data_numpy, label
