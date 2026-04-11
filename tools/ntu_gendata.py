"""NTU-RGB-D 数据转换脚本。"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

from numpy.lib.format import open_memmap

from utils.ntu_read_skeleton import read_xyz

training_subjects = [
    1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38
]
training_cameras = [2, 3]
max_body = 2
num_joint = 25
max_frame = 300
toolbar_width = 30


def print_toolbar(rate: float, annotation: str = "") -> None:
    """打印简易进度条。"""
    sys.stdout.write(f"{annotation}[")
    for i in range(toolbar_width):
        if i * 1.0 / toolbar_width > rate:
            sys.stdout.write(" ")
        else:
            sys.stdout.write("-")
        sys.stdout.flush()
    sys.stdout.write("]\r")


def end_toolbar() -> None:
    """结束进度条输出。"""
    sys.stdout.write("\n")


def gendata(
    data_path: str,
    out_path: str,
    ignored_sample_path: str | None = None,
    benchmark: str = "xview",
    part: str = "eval",
) -> None:
    """生成 NTU 数据文件。"""
    if ignored_sample_path is not None:
        with open(ignored_sample_path, "r", encoding="utf-8") as file_obj:
            ignored_samples = [line.strip() + ".skeleton" for line in file_obj.readlines()]
    else:
        ignored_samples = []

    sample_name: list[str] = []
    sample_label: list[int] = []
    for filename in os.listdir(data_path):
        if filename in ignored_samples:
            continue
        action_class = int(filename[filename.find("A") + 1 : filename.find("A") + 4])
        subject_id = int(filename[filename.find("P") + 1 : filename.find("P") + 4])
        camera_id = int(filename[filename.find("C") + 1 : filename.find("C") + 4])

        if benchmark == "xview":
            istraining = camera_id in training_cameras
        elif benchmark == "xsub":
            istraining = subject_id in training_subjects
        else:
            raise ValueError()

        if part == "train":
            issample = istraining
        elif part == "val":
            issample = not istraining
        else:
            raise ValueError()

        if issample:
            sample_name.append(filename)
            sample_label.append(action_class - 1)

    with open(f"{out_path}/{part}_label.pkl", "wb") as file_obj:
        pickle.dump((sample_name, list(sample_label)), file_obj)

    fp = open_memmap(
        f"{out_path}/{part}_data.npy",
        dtype="float32",
        mode="w+",
        shape=(len(sample_label), 3, max_frame, num_joint, max_body),
    )

    for i, sample in enumerate(sample_name):
        print_toolbar(
            i * 1.0 / len(sample_label),
            "({:>5}/{:<5}) Processing {:>5}-{:<5} data: ".format(i + 1, len(sample_name), benchmark, part),
        )
        data = read_xyz(os.path.join(data_path, sample), max_body=max_body, num_joint=num_joint)
        fp[i, :, 0 : data.shape[1], :, :] = data
    end_toolbar()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="NTU-RGB-D Data Converter.")
    parser.add_argument("--data_path", default="data/NTU-RGB-D/nturgb+d_skeletons")
    parser.add_argument(
        "--ignored_sample_path",
        default="resource/NTU-RGB-D/samples_with_missing_skeletons.txt",
    )
    parser.add_argument("--out_folder", default="data/NTU-RGB-D")

    arg = parser.parse_args()
    for benchmark in ["xsub", "xview"]:
        for part in ["train", "val"]:
            out_path = os.path.join(arg.out_folder, benchmark)
            if not os.path.exists(out_path):
                os.makedirs(out_path)
            gendata(arg.data_path, out_path, arg.ignored_sample_path, benchmark=benchmark, part=part)
