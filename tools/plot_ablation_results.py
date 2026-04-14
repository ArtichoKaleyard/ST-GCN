#!/usr/bin/env python3
"""绘制 NTU xsub 第一轮消融实验结果。"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / "temp" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import yaml

TRAIN_EPOCH_PATTERN = re.compile(r"Training epoch:\s*(\d+)")
MEAN_LOSS_PATTERN = re.compile(r"\bmean_loss:\s*([0-9]*\.?[0-9]+)")
EVAL_EPOCH_PATTERN = re.compile(r"Eval epoch:\s*(\d+)")
TOPK_PATTERN = re.compile(r"Top(\d+):\s*([0-9]*\.?[0-9]+)%")


@dataclass
class AblationRun:
    """存放单组消融实验的核心指标。"""

    label: str
    work_dir: Path
    strategy: str
    edge_importance_weighting: bool
    total_epochs: int
    train_epochs: list[int]
    train_mean_loss: list[float]
    eval_epochs: list[int]
    eval_top1: list[float]
    eval_top5: list[float]

    @property
    def final_top1(self) -> float:
        return self.eval_top1[-1]

    @property
    def final_top5(self) -> float:
        return self.eval_top5[-1]

    @property
    def best_top1(self) -> float:
        return max(self.eval_top1)

    @property
    def best_epoch(self) -> int:
        return self.eval_epochs[int(np.argmax(self.eval_top1))]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="绘制第一轮消融实验结果")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        nargs=2,
        metavar=("LABEL", "WORK_DIR"),
        help="指定一组实验，格式为 --run 标签 工作目录；默认读取 4 组第一轮消融。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "work_dir" / "figures",
        help="图像输出目录。",
    )
    return parser.parse_args()


def apply_style() -> None:
    """应用 SciencePlots 学术风格。"""
    plt.style.use(["science", "grid", "no-latex"])
    mpl.rcParams.update(
        {
            "figure.dpi": 240,
            "savefig.dpi": 400,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "axes.edgecolor": "#2F2F2F",
            "grid.color": "#D6D6D6",
            "grid.alpha": 0.55,
            "grid.linewidth": 0.7,
            "grid.linestyle": "--",
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9.4,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.0,
            "xtick.labelsize": 8.6,
            "ytick.labelsize": 8.6,
            "legend.fontsize": 8.4,
            "legend.frameon": False,
        }
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件。"""
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def parse_run(label: str, work_dir: Path) -> AblationRun:
    """解析单个工作目录下的训练和评估曲线。"""
    config_path = work_dir / "config.yaml"
    log_path = work_dir / "log.txt"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"未找到日志文件: {log_path}")

    config = load_yaml(config_path)
    model_args = config.get("model_args", {})
    graph_args = model_args.get("graph_args", {})

    train_epochs: list[int] = []
    train_mean_loss: list[float] = []
    eval_epochs: list[int] = []
    eval_top1: list[float] = []
    eval_top5: list[float] = []

    current_train_epoch: int | None = None
    current_eval_epoch: int | None = None
    pending_topk: dict[int, float] = {}

    with log_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            train_match = TRAIN_EPOCH_PATTERN.search(line)
            if train_match:
                current_train_epoch = int(train_match.group(1)) + 1
                continue

            mean_loss_match = MEAN_LOSS_PATTERN.search(line)
            if mean_loss_match and current_train_epoch is not None:
                train_epochs.append(current_train_epoch)
                train_mean_loss.append(float(mean_loss_match.group(1)))
                current_train_epoch = None
                continue

            eval_match = EVAL_EPOCH_PATTERN.search(line)
            if eval_match:
                current_eval_epoch = int(eval_match.group(1)) + 1
                pending_topk = {}
                continue

            topk_match = TOPK_PATTERN.search(line)
            if topk_match and current_eval_epoch is not None:
                pending_topk[int(topk_match.group(1))] = float(topk_match.group(2))
                if 1 in pending_topk and 5 in pending_topk:
                    eval_epochs.append(current_eval_epoch)
                    eval_top1.append(pending_topk[1])
                    eval_top5.append(pending_topk[5])
                    current_eval_epoch = None
                    pending_topk = {}

    if not train_epochs or not train_mean_loss:
        raise ValueError(f"未从 {log_path} 解析出训练 loss 曲线。")
    if not eval_epochs or not eval_top1 or not eval_top5:
        raise ValueError(f"未从 {log_path} 解析出验证 Top-k 曲线。")

    return AblationRun(
        label=label,
        work_dir=work_dir,
        strategy=str(graph_args.get("strategy", "")),
        edge_importance_weighting=bool(model_args.get("edge_importance_weighting", False)),
        total_epochs=int(config.get("num_epoch", 0)),
        train_epochs=train_epochs,
        train_mean_loss=train_mean_loss,
        eval_epochs=eval_epochs,
        eval_top1=eval_top1,
        eval_top5=eval_top5,
    )


def build_default_runs() -> list[tuple[str, Path]]:
    """返回第一轮消融的默认 4 组实验。"""
    root = ROOT_DIR / "work_dir" / "ablation" / "ntu-xsub"
    return [
        ("Uniform / No Imp.", root / "uniform_noimp"),
        ("Distance / No Imp.", root / "distance_noimp"),
        ("Spatial / No Imp.", root / "spatial_noimp"),
        ("Spatial / Imp.", root / "spatial_imp"),
    ]


def plot_runs(runs: list[AblationRun], output_dir: Path) -> list[Path]:
    """绘制消融对比图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = ["#A63D40", "#E09F3E", "#2A6F97", "#3A7D44"]
    markers = ["o", "s", "^", "D"]

    fig, axes = plt.subplot_mosaic(
        [["top1", "top5"], ["loss", "summary"]],
        figsize=(8.4, 6.2),
        gridspec_kw={"height_ratios": [1.05, 1.0], "width_ratios": [1.05, 0.95]},
        constrained_layout=True,
    )
    ordered_runs = sorted(runs, key=lambda run: run.final_top1)
    summary_positions = np.arange(len(ordered_runs))

    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        axes["top1"].plot(
            run.eval_epochs,
            run.eval_top1,
            color=color,
            linewidth=1.9,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.1,
            label=run.label,
        )
        axes["top5"].plot(
            run.eval_epochs,
            run.eval_top5,
            color=color,
            linewidth=1.8,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label=run.label,
        )
        axes["loss"].plot(
            run.train_epochs,
            run.train_mean_loss,
            color=color,
            linewidth=1.8,
            marker=marker,
            markersize=3.5,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label=run.label,
        )

    summary_colors = [colors[runs.index(run) % len(colors)] for run in ordered_runs]
    axes["summary"].barh(
        summary_positions + 0.16,
        [run.best_top1 for run in ordered_runs],
        height=0.28,
        color=summary_colors,
        alpha=0.38,
        label="Best",
    )
    axes["summary"].barh(
        summary_positions - 0.16,
        [run.final_top1 for run in ordered_runs],
        height=0.28,
        color=summary_colors,
        alpha=0.92,
        label="Final",
    )
    axes["summary"].set_yticks(summary_positions)
    axes["summary"].set_yticklabels([run.label for run in ordered_runs])
    axes["summary"].invert_yaxis()

    summary_xmax = max(run.best_top1 for run in runs) + 2.0
    for pos, run in zip(summary_positions, ordered_runs, strict=True):
        axes["summary"].text(
            run.best_top1 + 0.18,
            pos + 0.16,
            f"{run.best_top1:.2f} @ {run.best_epoch}",
            va="center",
            ha="left",
            fontsize=8.0,
            color="#4A4A4A",
        )
        axes["summary"].text(
            run.final_top1 + 0.18,
            pos - 0.16,
            f"{run.final_top1:.2f}",
            va="center",
            ha="left",
            fontsize=8.0,
            color="#2F2F2F",
        )

    axes["top1"].set(title="Validation Top-1", xlabel="Epoch", ylabel="Accuracy (%)")
    axes["loss"].set(title="Training Mean Loss", xlabel="Epoch", ylabel="Cross-Entropy Loss")
    axes["top5"].set(title="Validation Top-5", xlabel="Epoch", ylabel="Accuracy (%)")
    axes["summary"].set(title="Best / Final Top-1", xlabel="Accuracy (%)")

    axes["top1"].legend(loc="lower right", fontsize=8.2, handlelength=1.5)
    axes["loss"].legend(loc="upper right", ncol=2, handlelength=1.8)
    axes["top5"].legend(loc="lower right", fontsize=8.2)
    axes["top1"].grid(axis="x", visible=False)
    axes["top5"].grid(axis="x", visible=False)
    axes["summary"].grid(axis="y", visible=False)

    eval_epochs = sorted({epoch for run in runs for epoch in run.eval_epochs})
    axes["top1"].set_xticks(eval_epochs)
    axes["top5"].set_xticks(eval_epochs)
    max_epoch = max(run.total_epochs for run in runs)
    axes["loss"].set_xlim(1, max_epoch)
    axes["top1"].set_xlim(min(eval_epochs) - 1, max_epoch + 1)
    axes["top5"].set_xlim(min(eval_epochs) - 1, max_epoch + 1)
    axes["top1"].set_ylim(
        min(min(run.eval_top1) for run in runs) - 2.0,
        max(max(run.eval_top1) for run in runs) + 1.2,
    )
    axes["top5"].set_ylim(
        min(min(run.eval_top5) for run in runs) - 1.2,
        max(max(run.eval_top5) for run in runs) + 0.6,
    )
    axes["summary"].set_xlim(
        min(run.final_top1 for run in runs) - 3.0,
        summary_xmax,
    )

    best_run = max(runs, key=lambda run: run.best_top1)
    worst_run = min(runs, key=lambda run: run.final_top1)
    fig.suptitle(
        (
            "NTU RGB+D xsub First-Round Ablation: "
            f"{best_run.label} best {best_run.best_top1:.2f}, "
            f"{worst_run.label} final {worst_run.final_top1:.2f}"
        ),
        x=0.02,
        y=1.01,
        ha="left",
        va="bottom",
        fontsize=11.0,
    )
    axes["top1"].text(
        0.01,
        0.96,
        "40 epochs, AMP on, batch size 16, step = [10, 20, 30]",
        transform=axes["top1"].transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#5F6368",
    )
    axes["summary"].text(
        0.99,
        0.96,
        "dark bars = final, light bars = best",
        transform=axes["summary"].transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color="#5F6368",
    )

    png_path = output_dir / "ntu_xsub_ablation_round1.png"
    pdf_path = output_dir / "ntu_xsub_ablation_round1.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    apply_style()
    run_specs = args.runs if args.runs is not None else build_default_runs()
    runs = [parse_run(label, Path(work_dir)) for label, work_dir in run_specs]
    outputs = plot_runs(runs, args.output_dir)
    print("已导出图像：")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
