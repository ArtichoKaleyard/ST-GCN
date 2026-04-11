"""Kinetics-skeleton 数据转换脚本。"""

from __future__ import annotations

import argparse
import os
import pickle
import sys

from numpy.lib.format import open_memmap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from feeder.feeder_kinetics import Feeder_kinetics

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
    label_path: str,
    data_out_path: str,
    label_out_path: str,
    num_person_in: int = 5,
    num_person_out: int = 2,
    max_frame: int = 300,
) -> None:
    """生成 Kinetics 数据文件。"""
    feeder = Feeder_kinetics(
        data_path=data_path,
        label_path=label_path,
        num_person_in=num_person_in,
        num_person_out=num_person_out,
        window_size=max_frame,
    )

    sample_name = feeder.sample_name
    sample_label: list[int] = []
    fp = open_memmap(
        data_out_path,
        dtype="float32",
        mode="w+",
        shape=(len(sample_name), 3, max_frame, 18, num_person_out),
    )

    for i, _sample in enumerate(sample_name):
        data, label = feeder[i]
        print_toolbar(
            i * 1.0 / len(sample_name),
            "({:>5}/{:<5}) Processing data: ".format(i + 1, len(sample_name)),
        )
        fp[i, :, 0 : data.shape[1], :, :] = data
        sample_label.append(label)

    with open(label_out_path, "wb") as file_obj:
        pickle.dump((sample_name, list(sample_label)), file_obj)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kinetics-skeleton Data Converter.")
    parser.add_argument("--data_path", default="data/Kinetics/kinetics-skeleton")
    parser.add_argument("--out_folder", default="data/Kinetics/kinetics-skeleton")
    arg = parser.parse_args()

    for part in ["train", "val"]:
        data_path = f"{arg.data_path}/kinetics_{part}"
        label_path = f"{arg.data_path}/kinetics_{part}_label.json"
        data_out_path = f"{arg.out_folder}/{part}_data.npy"
        label_out_path = f"{arg.out_folder}/{part}_label.pkl"

        if not os.path.exists(arg.out_folder):
            os.makedirs(arg.out_folder)
        gendata(data_path, label_path, data_out_path, label_out_path)
