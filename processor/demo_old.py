#!/usr/bin/env python
"""旧版离线 demo。"""

from __future__ import annotations

import argparse
import json
import os
import shutil

import skvideo.io
import torch

import tools.utils as utils

from .io import IO


class Demo(IO):
    """Demo for Skeleton-based Action Recognition。"""

    def start(self) -> None:
        openpose = f"{self.arg.openpose}/examples/openpose/openpose.bin"
        video_name = self.arg.video.split("/")[-1].split(".")[0]
        output_snippets_dir = f"./data/openpose_estimation/snippets/{video_name}"
        output_sequence_dir = "./data/openpose_estimation/data"
        output_sequence_path = f"{output_sequence_dir}/{video_name}.json"
        output_result_dir = self.arg.output_dir
        output_result_path = f"{output_result_dir}/{video_name}.mp4"
        label_name_path = "./resource/kinetics_skeleton/label_name.txt"
        with open(label_name_path, encoding="utf-8") as file_obj:
            label_name = [line.rstrip() for line in file_obj.readlines()]

        openpose_args = dict(
            video=self.arg.video,
            write_json=output_snippets_dir,
            display=0,
            render_pose=0,
            model_pose="COCO",
        )
        command_line = openpose + " "
        command_line += " ".join(f"--{key} {value}" for key, value in openpose_args.items())
        shutil.rmtree(output_snippets_dir, ignore_errors=True)
        os.makedirs(output_snippets_dir)
        os.system(command_line)

        video = utils.video.get_video_frames(self.arg.video)
        height, width, _ = video[0].shape
        video_info = utils.openpose.json_pack(output_snippets_dir, video_name, width, height)
        if not os.path.exists(output_sequence_dir):
            os.makedirs(output_sequence_dir)
        with open(output_sequence_path, "w", encoding="utf-8") as outfile:
            json.dump(video_info, outfile)
        if len(video_info["data"]) == 0:
            print("Can not find pose estimation results.")
            return
        print("Pose estimation complete.")

        pose, _ = utils.video.video_info_parsing(video_info)
        data = torch.from_numpy(pose).unsqueeze(0).float().to(self.dev).detach()

        print("\nNetwork forwad...")
        self.model.eval()
        output, feature = self.model.extract_feature(data)
        output = output[0]
        feature = feature[0]
        intensity = (feature * feature).sum(dim=0) ** 0.5
        intensity = intensity.cpu().detach().numpy()
        label = output.sum(dim=3).sum(dim=2).sum(dim=1).argmax(dim=0)
        print(f"Prediction result: {label_name[label]}")
        print("Done.")

        print("\nVisualization...")
        label_sequence = output.sum(dim=2).argmax(dim=0)
        label_name_sequence = [[label_name[p] for p in labels] for labels in label_sequence]
        edge = self.model.graph.edge
        images = utils.visualization.stgcn_visualize(
            pose, edge, intensity, video, label_name[label], label_name_sequence, self.arg.height
        )
        print("Done.")

        print("\nSaving...")
        if not os.path.exists(output_result_dir):
            os.makedirs(output_result_dir)
        writer = skvideo.io.FFmpegWriter(output_result_path, outputdict={"-b": "300000000"})
        for image in images:
            writer.writeFrame(image)
        writer.close()
        print(f"The Demo result has been saved in {output_result_path}.")

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
        parser.add_argument("--openpose", default="3dparty/openpose/build", help="Path to openpose")
        parser.add_argument("--output_dir", default="./data/demo_result", help="Path to save results")
        parser.add_argument("--height", default=1080, type=int)
        parser.set_defaults(config="./config/st_gcn/kinetics-skeleton/demo_old.yaml")
        parser.set_defaults(print_log=False)

        return parser
