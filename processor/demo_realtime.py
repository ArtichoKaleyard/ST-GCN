#!/usr/bin/env python
"""实时视频 demo。"""

from __future__ import annotations

import argparse
import sys
import time

import cv2
import numpy as np
import torch

import tools.utils as utils

from .io import IO
from .demo_offline import naive_pose_tracker


class DemoRealtime(IO):
    """实时动作识别 demo。"""

    def start(self) -> None:
        if self.arg.openpose is not None:
            sys.path.append(f"{self.arg.openpose}/python")
            sys.path.append(f"{self.arg.openpose}/build/python")
        try:
            from openpose import pyopenpose as op
        except Exception:
            print("Can not find Openpose Python API.")
            return

        label_name_path = "./resource/kinetics_skeleton/label_name.txt"
        with open(label_name_path, encoding="utf-8") as file_obj:
            self.label_name = [line.rstrip() for line in file_obj.readlines()]

        op_wrapper = op.WrapperPython()
        params = dict(model_folder="./models", model_pose="COCO")
        op_wrapper.configure(params)
        op_wrapper.start()
        self.model.eval()
        pose_tracker = naive_pose_tracker()

        if self.arg.video == "camera_source":
            video_capture = cv2.VideoCapture(0)
        else:
            video_capture = cv2.VideoCapture(self.arg.video)

        start_time = time.time()
        frame_index = 0
        while True:
            tic = time.time()
            _ret, orig_image = video_capture.read()
            if orig_image is None:
                break
            source_H, source_W, _ = orig_image.shape
            orig_image = cv2.resize(orig_image, (256 * source_W // source_H, 256))
            H, W, _ = orig_image.shape

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

            if self.arg.video == "camera_source":
                frame_index = int((time.time() - start_time) * self.arg.fps)
            else:
                frame_index += 1
            pose_tracker.update(multi_pose, frame_index)
            data_numpy = pose_tracker.get_skeleton_sequence()
            data = torch.from_numpy(data_numpy).unsqueeze(0).float().to(self.dev).detach()

            voting_label_name, video_label_name, _output, intensity = self.predict(data)

            app_fps = 1 / (time.time() - tic)
            image = self.render(
                data_numpy, voting_label_name, video_label_name, intensity, orig_image, app_fps
            )
            cv2.imshow("ST-GCN", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    def predict(self, data: torch.Tensor) -> tuple[str, list[list[str]], torch.Tensor, np.ndarray]:
        """执行预测并返回可视化信息。"""
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

    def render(
        self,
        data_numpy: np.ndarray,
        voting_label_name: str,
        video_label_name: list[list[str]],
        intensity: np.ndarray,
        orig_image: np.ndarray,
        fps: float = 0,
    ) -> np.ndarray:
        """渲染当前帧。"""
        images = utils.visualization.stgcn_visualize(
            data_numpy[:, [-1]],
            self.model.graph.edge,
            intensity[[-1]],
            [orig_image],
            voting_label_name,
            [video_label_name[-1]],
            self.arg.height,
            fps=fps,
        )
        image = next(images)
        return image.astype(np.uint8)

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
        parser.set_defaults(config="./config/st_gcn/kinetics-skeleton/demo_realtime.yaml")
        parser.set_defaults(print_log=False)

        return parser
