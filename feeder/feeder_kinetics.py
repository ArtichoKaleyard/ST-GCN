"""Kinetics-skeleton 数据集 feeder。"""

from __future__ import annotations

import json
import os

import numpy as np
import torch

from . import tools


class Feeder_kinetics(torch.utils.data.Dataset):
    """Kinetics-skeleton 数据集 feeder。

    Args:
        data_path: 按样本拆分的骨架 JSON 目录。
        label_path: Kinetics 标签 JSON。
        ignore_empty_sample: 是否过滤 `has_skeleton == False` 的样本。
        random_choose: 是否随机裁剪时间窗。
        random_shift: 是否在时间维随机平移有效帧。
        random_move: 是否做连续随机仿射扰动。
        window_size: 输出时间窗长度。
        pose_matching: 是否在相邻帧间执行人体匹配。
        num_person_in: 单个样本最多读取多少个人体实例。
        num_person_out: 输出时最多保留多少个人体实例。
        debug: 是否只取极少样本做快速调试。
    """

    def __init__(
        self,
        data_path: str,
        label_path: str,
        ignore_empty_sample: bool = True,
        random_choose: bool = False,
        random_shift: bool = False,
        random_move: bool = False,
        window_size: int = -1,
        pose_matching: bool = False,
        num_person_in: int = 5,
        num_person_out: int = 2,
        debug: bool = False,
    ) -> None:
        self.debug = debug
        self.data_path = data_path
        self.label_path = label_path
        self.random_choose = random_choose
        self.random_shift = random_shift
        self.random_move = random_move
        self.window_size = window_size
        self.num_person_in = num_person_in
        self.num_person_out = num_person_out
        self.pose_matching = pose_matching
        self.ignore_empty_sample = ignore_empty_sample

        self.load_data()

    def load_data(self) -> None:
        """加载文件列表与标签信息。"""
        self.sample_name = os.listdir(self.data_path)

        if self.debug:
            self.sample_name = self.sample_name[0:2]

        with open(self.label_path, encoding="utf-8") as file_obj:
            label_info = json.load(file_obj)

        sample_id = [name.split(".")[0] for name in self.sample_name]
        self.label = np.array([label_info[sample]["label_index"] for sample in sample_id])
        has_skeleton = np.array([label_info[sample]["has_skeleton"] for sample in sample_id])

        # Kinetics 标注里可能存在没有骨架的样本；官方实现默认直接过滤掉。
        if self.ignore_empty_sample:
            self.sample_name = [s for h, s in zip(has_skeleton, self.sample_name) if h]
            self.label = self.label[has_skeleton]

        self.N = len(self.sample_name)
        self.C = 3
        self.T = 300
        self.V = 18
        self.M = self.num_person_out

    def __len__(self) -> int:
        return len(self.sample_name)

    def __iter__(self) -> "Feeder_kinetics":
        return self

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        """返回单个 kinetics 样本。"""
        sample_name = self.sample_name[index]
        sample_path = os.path.join(self.data_path, sample_name)
        with open(sample_path, "r", encoding="utf-8") as file_obj:
            video_info = json.load(file_obj)

        data_numpy = np.zeros((self.C, self.T, self.V, self.num_person_in))
        for frame_info in video_info["data"]:
            frame_index = frame_info["frame_index"]
            for m, skeleton_info in enumerate(frame_info["skeleton"]):
                if m >= self.num_person_in:
                    break
                pose = skeleton_info["pose"]
                score = skeleton_info["score"]
                data_numpy[0, frame_index, :, m] = pose[0::2]
                data_numpy[1, frame_index, :, m] = pose[1::2]
                data_numpy[2, frame_index, :, m] = score

        # 与官方预处理一致：先把 xy 中心化到 [-0.5, 0.5] 左右，再把缺失点清零。
        data_numpy[0:2] = data_numpy[0:2] - 0.5
        data_numpy[0][data_numpy[2] == 0] = 0
        data_numpy[1][data_numpy[2] == 0] = 0

        label = video_info["label_index"]
        assert self.label[index] == label

        if self.random_shift:
            data_numpy = tools.random_shift(data_numpy)
        if self.random_choose:
            data_numpy = tools.random_choose(data_numpy, self.window_size)
        elif self.window_size > 0:
            data_numpy = tools.auto_pading(data_numpy, self.window_size)
        if self.random_move:
            data_numpy = tools.random_move(data_numpy)

        # 每一帧按 skeleton score 排序，只保留分数最高的若干人体实例。
        sort_index = (-data_numpy[2, :, :, :].sum(axis=1)).argsort(axis=1)
        for t, sort_order in enumerate(sort_index):
            data_numpy[:, t, :, :] = data_numpy[:, t, :, sort_order].transpose((1, 2, 0))
        data_numpy = data_numpy[:, :, :, 0 : self.num_person_out]

        # 可选地在时间维上重排人体轨迹，使同一人尽量保持实例槽位一致。
        if self.pose_matching:
            data_numpy = tools.openpose_match(data_numpy)

        return data_numpy, label

    def top_k(self, score: np.ndarray, top_k: int) -> float:
        """计算 top-k 准确率。"""
        assert all(self.label >= 0)
        rank = score.argsort()
        hit_top_k = [label in rank[i, -top_k:] for i, label in enumerate(self.label)]
        return sum(hit_top_k) * 1.0 / len(hit_top_k)

    def top_k_by_category(self, score: np.ndarray, top_k: int) -> list[float]:
        """按类别统计 top-k。"""
        assert all(self.label >= 0)
        return tools.top_k_by_category(self.label, score, top_k)

    def calculate_recall_precision(self, score: np.ndarray) -> tuple[list[float], list[float]]:
        """计算召回率与精确率。"""
        assert all(self.label >= 0)
        return tools.calculate_recall_precision(self.label, score)
