#!/usr/bin/env python
"""离线视频 demo。"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import torch

import tools.utils as utils

from .io import IO


class DemoOffline(IO):
    """离线动作识别 demo。"""

    def start(self) -> None:
        label_name_path = "./resource/kinetics_skeleton/label_name.txt"
        with open(label_name_path, encoding="utf-8") as file_obj:
            self.label_name = [line.rstrip() for line in file_obj.readlines()]

        video, data_numpy = self.pose_estimation()
        data = torch.from_numpy(data_numpy).unsqueeze(0).float().to(self.dev).detach()

        voting_label_name, video_label_name, _output, intensity = self.predict(data)
        images = self.render_video(data_numpy, voting_label_name, video_label_name, intensity, video)

        for image in images:
            image = image.astype(np.uint8)
            cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def predict(self, data: torch.Tensor) -> tuple[str, list[list[str]], torch.Tensor, np.ndarray]:
        """执行预测并返回可视化需要的信息。"""
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0) ** 0.5
        intensity = intensity.cpu().detach().numpy()

        voting_label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        voting_label_name = self.label_name[voting_label]

        num_person = output.size(3)
        num_frame = output.size(1)
        video_label_name: list[list[str]] = []
        for t in range(num_frame):
            frame_label_name: list[str] = []
            for m in range(num_person):
                person_label = output[:, t, :, m].sum(dim=1).argmax(dim=0)
                frame_label_name.append(self.label_name[person_label])
            video_label_name.append(frame_label_name)
        return voting_label_name, video_label_name, output, intensity

    def render_video(
        self,
        data_numpy: np.ndarray,
        voting_label_name: str,
        video_label_name: list[list[str]],
        intensity: np.ndarray,
        video: list[np.ndarray],
    ):
        """渲染可视化视频。"""
        return utils.visualization.stgcn_visualize(
            data_numpy,
            self.model.graph.edge,
            intensity,
            video,
            voting_label_name,
            video_label_name,
            self.arg.height,
        )

    def pose_estimation(self) -> tuple[list[np.ndarray], np.ndarray]:
        """执行 OpenPose 与简单跟踪。"""
        if self.arg.openpose is not None:
            sys.path.append(f"{self.arg.openpose}/python")
            sys.path.append(f"{self.arg.openpose}/build/python")
        try:
            from openpose import pyopenpose as op
        except Exception:
            print("Can not find Openpose Python API.")
            return None

        op_wrapper = op.WrapperPython()
        params = dict(model_folder="./models", model_pose="COCO")
        op_wrapper.configure(params)
        op_wrapper.start()
        self.model.eval()
        video_capture = cv2.VideoCapture(self.arg.video)
        video_length = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
        pose_tracker = naive_pose_tracker(data_frame=video_length)

        frame_index = 0
        video: list[np.ndarray] = []
        while True:
            _ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape
            video.append(orig_image)

            datum = op.Datum()
            datum.cvInputData = orig_image
            op_wrapper.emplaceAndPop([datum])
            multi_pose = datum.poseKeypoints
            if len(multi_pose.shape) != 3:
                continue

            multi_pose[:, :, 0] = multi_pose[:, :, 0] / W
            multi_pose[:, :, 1] = multi_pose[:, :, 1] / H
            multi_pose[:, :, 0:2] = multi_pose[:, :, 0:2] - 0.5
            multi_pose[:, :, 0][multi_pose[:, :, 2] == 0] = 0
            multi_pose[:, :, 1][multi_pose[:, :, 2] == 0] = 0

            pose_tracker.update(multi_pose, frame_index)
            frame_index += 1
            print(f"Pose estimation ({frame_index}/{video_length}).")

        data_numpy = pose_tracker.get_skeleton_sequence()
        return video, data_numpy

    @staticmethod
    def get_parser(add_help: bool = False) -> argparse.ArgumentParser:
        """获取参数解析器。"""
        parent_parser = IO.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description="Demo for Spatial Temporal Graph Convolution Network",
        )

        parser.add_argument("--video", default="./resource/media/skateboarding.mp4", help="Path to video")
        parser.add_argument("--openpose", default=None, help="Path to openpose")
        parser.add_argument("--model_input_frame", default=128, type=int)
        parser.add_argument("--model_fps", default=30, type=int)
        parser.add_argument("--height", default=1080, type=int, help="height of frame in the output video.")
        parser.set_defaults(config="./config/st_gcn/kinetics-skeleton/demo_offline.yaml")
        parser.set_defaults(print_log=False)

        return parser


class naive_pose_tracker:
    """简单人体跟踪器。"""

    def __init__(self, data_frame: int = 128, num_joint: int = 18, max_frame_dis: float = np.inf):
        self.data_frame = data_frame
        self.num_joint = num_joint
        self.max_frame_dis = max_frame_dis
        self.latest_frame = 0
        self.trace_info: list[tuple[np.ndarray, int]] = []

    def update(self, multi_pose: np.ndarray, current_frame: int) -> None:
        if current_frame <= self.latest_frame:
            return
        if len(multi_pose.shape) != 3:
            return

        score_order = (-multi_pose[:, :, 2].sum(axis=1)).argsort(axis=0)
        for pose in multi_pose[score_order]:
            matching_trace = None
            matching_dis = None
            for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
                if current_frame <= latest_frame:
                    continue
                mean_dis, is_close = self.get_dis(trace, pose)
                if is_close:
                    if matching_trace is None or matching_dis > mean_dis:
                        matching_trace = trace_index
                        matching_dis = mean_dis

            if matching_trace is not None:
                trace, latest_frame = self.trace_info[matching_trace]
                pad_mode = "interp" if latest_frame == self.latest_frame else "zero"
                pad = current_frame - latest_frame - 1
                new_trace = self.cat_pose(trace, pose, pad, pad_mode)
                self.trace_info[matching_trace] = (new_trace, current_frame)
            else:
                self.trace_info.append((np.array([pose]), current_frame))

        self.latest_frame = current_frame

    def get_skeleton_sequence(self) -> np.ndarray:
        """生成 ST-GCN 输入序列。"""
        valid_trace_index = []
        for trace_index, (_trace, latest_frame) in enumerate(self.trace_info):
            if self.latest_frame - latest_frame < self.data_frame:
                valid_trace_index.append(trace_index)
        self.trace_info = [self.trace_info[index] for index in valid_trace_index]

        num_trace = len(self.trace_info)
        if num_trace == 0:
            return None

        data = np.zeros((3, self.data_frame, self.num_joint, num_trace))
        for trace_index, (trace, latest_frame) in enumerate(self.trace_info):
            end = self.data_frame - (self.latest_frame - latest_frame)
            d = trace[-end:]
            beg = end - len(d)
            data[:, beg:end, :, trace_index] = d.transpose((2, 0, 1))

        sort_index = (-data[2, :, :, :].sum(axis=1).sum(axis=0)).argsort(axis=0)
        data = data[:, :, :, sort_index]
        return data[:, :, :, 0:2]

    def get_dis(self, trace: np.ndarray, pose: np.ndarray) -> tuple[float, bool]:
        """计算轨迹末帧与当前姿态的距离。"""
        last_pose_xy = trace[-1, :, 0:2]
        curr_pose_xy = pose[:, 0:2]

        mean_dis = ((((last_pose_xy - curr_pose_xy) ** 2).sum(axis=1)) ** 0.5).mean()
        wh = last_pose_xy.max(axis=0) - last_pose_xy.min(axis=0)
        scale = (wh[0] ** 2 + wh[1] ** 2) ** 0.5 + 0.0001
        is_close = mean_dis < scale * self.max_frame_dis
        return mean_dis, is_close

    def cat_pose(self, trace: np.ndarray, pose: np.ndarray, pad: int, pad_mode: str) -> np.ndarray:
        """将新姿态拼接到已有轨迹。"""
        if pad != 0:
            if pad_mode == "zero":
                trace = np.concatenate((trace, np.zeros((pad, self.num_joint, 3))), 0)
            elif pad_mode == "interp":
                last_pose = trace[-1]
                coeff = [(p + 1) / (pad + 1) for p in range(pad)]
                interp_pose = [(1 - c) * last_pose + c * pose for c in coeff]
                trace = np.concatenate((trace, interp_pose), 0)
        return np.concatenate((trace, [pose]), 0)
