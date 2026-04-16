#!/usr/bin/env python3
"""对比两次 ST-GCN 正式训练结果并导出学术风格图表。

该脚本面向当前仓库的 `work_dir` 训练产物，读取 `config.yaml` 与 `log.txt`，
兼容旧版时间戳日志和当前 Herald 日志，统一抽取以下曲线：

- 每个 epoch 的训练均值 loss
- 每次评估点的验证 Top-1 / Top-5
- 每个 epoch 的学习率

输出采用 Matplotlib 学术排版，不手工计算子图坐标，默认通过
`subplot_mosaic(..., layout="constrained")` 组织版式。
"""

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
ITER_PATTERN = re.compile(
    r"Iter\s+(\d+)\s+Done\.\s+\|\s+loss:\s*([0-9]*\.?[0-9]+)\s+\|\s+lr:\s*([0-9]*\.?[0-9]+)"
)
EVAL_EPOCH_PATTERN = re.compile(r"Eval epoch:\s*(\d+)")
TOPK_PATTERN = re.compile(r"Top(\d+):\s*([0-9]*\.?[0-9]+)%")


@dataclass
class RunMetrics:
    """存放单次训练运行的关键曲线。"""

    label: str
    work_dir: Path
    config_path: str
    lr_steps: list[int]
    total_epochs: int
    iter_steps: list[int]
    iter_loss: list[float]
    iter_epoch_x: list[float]
    global_steps: list[int]
    train_epochs: list[int]
    train_mean_loss: list[float]
    epoch_lr: list[float]
    eval_epochs: list[int]
    eval_top1: list[float]
    eval_top5: list[float]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="对比两次训练结果并绘图")
    parser.add_argument(
        "--run",
        dest="runs",
        action="append",
        nargs=2,
        metavar=("LABEL", "WORK_DIR"),
        help="传入一组对比对象，格式为: --run 标签 工作目录；需提供两组。",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "work_dir" / "figures",
        help="图像输出目录。",
    )
    parser.add_argument(
        "--style",
        type=Path,
        default=None,
        help="保留兼容参数，当前默认使用 SciencePlots；若传入自定义 mplstyle 则会追加覆盖。",
    )
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件。"""
    with path.open("r", encoding="utf-8") as file_obj:
        return yaml.safe_load(file_obj)


def parse_run(label: str, work_dir: Path) -> RunMetrics:
    """解析单个工作目录下的训练与评估曲线。

    Args:
        label: 图中显示的运行标签。
        work_dir: 包含 `config.yaml` 与 `log.txt` 的工作目录。

    Returns:
        抽取后的曲线结构。

    Raises:
        FileNotFoundError: 目标日志或配置文件不存在。
        ValueError: 未能从日志中提取必要曲线。
    """

    config_path = work_dir / "config.yaml"
    log_path = work_dir / "log.txt"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"未找到日志文件: {log_path}")

    config = load_yaml(config_path)
    lr_steps = list(config.get("step", []))
    total_epochs = int(config.get("num_epoch", 0))

    train_epochs: list[int] = []
    train_mean_loss: list[float] = []
    epoch_lr: list[float] = []
    iter_steps: list[int] = []
    iter_loss: list[float] = []
    iter_epoch_x: list[float] = []
    global_steps: list[int] = []
    eval_epochs: list[int] = []
    eval_top1: list[float] = []
    eval_top5: list[float] = []

    current_train_epoch: int | None = None
    current_eval_epoch: int | None = None
    current_epoch_lr: float | None = None
    pending_topk: dict[int, float] = {}
    epoch_iter_steps: list[int] = []
    epoch_iter_loss: list[float] = []

    def flush_epoch_iter_trace(epoch_index: int | None) -> None:
        """把当前 epoch 的 iter 轨迹映射到连续 epoch 坐标。"""
        if epoch_index is None or not epoch_iter_steps:
            return
        point_count = len(epoch_iter_steps)
        if point_count == 1:
            local_positions = [0.5]
        else:
            local_positions = np.linspace(0.0, 1.0, point_count, endpoint=False).tolist()
        for pos, step_value, loss_value in zip(
            local_positions, epoch_iter_steps, epoch_iter_loss, strict=True
        ):
            iter_epoch_x.append(epoch_index + 1 + pos)
            iter_steps.append(step_value)
            iter_loss.append(loss_value)
            global_steps.append(len(global_steps) + 1)
        epoch_iter_steps.clear()
        epoch_iter_loss.clear()

    with log_path.open("r", encoding="utf-8") as file_obj:
        for raw_line in file_obj:
            line = raw_line.strip()
            if not line:
                continue

            train_match = TRAIN_EPOCH_PATTERN.search(line)
            if train_match:
                flush_epoch_iter_trace(current_train_epoch)
                current_train_epoch = int(train_match.group(1))
                current_epoch_lr = None
                continue

            iter_match = ITER_PATTERN.search(line)
            if iter_match and current_train_epoch is not None:
                if current_epoch_lr is None:
                    current_epoch_lr = float(iter_match.group(3))
                epoch_iter_steps.append(int(iter_match.group(1)))
                epoch_iter_loss.append(float(iter_match.group(2)))
                continue

            mean_loss_match = MEAN_LOSS_PATTERN.search(line)
            if mean_loss_match and current_train_epoch is not None:
                flush_epoch_iter_trace(current_train_epoch)
                train_epochs.append(current_train_epoch + 1)
                train_mean_loss.append(float(mean_loss_match.group(1)))
                epoch_lr.append(current_epoch_lr if current_epoch_lr is not None else 0.0)
                current_train_epoch = None
                current_epoch_lr = None
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

    flush_epoch_iter_trace(current_train_epoch)

    if not iter_epoch_x or not iter_loss or not train_epochs or not train_mean_loss:
        raise ValueError(f"未从 {log_path} 解析出训练 loss 曲线。")
    if not eval_epochs or not eval_top1 or not eval_top5:
        raise ValueError(f"未从 {log_path} 解析出验证 Top-k 曲线。")

    return RunMetrics(
        label=label,
        work_dir=work_dir,
        config_path=str(config.get("config", "")),
        lr_steps=lr_steps,
        total_epochs=total_epochs,
        iter_steps=iter_steps,
        iter_loss=iter_loss,
        iter_epoch_x=iter_epoch_x,
        global_steps=global_steps,
        train_epochs=train_epochs,
        train_mean_loss=train_mean_loss,
        epoch_lr=epoch_lr,
        eval_epochs=eval_epochs,
        eval_top1=eval_top1,
        eval_top5=eval_top5,
    )


def apply_style(style_path: Path | None = None) -> None:
    """应用 SciencePlots 论文风格，并允许额外 mplstyle 覆盖。"""
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
            "font.size": 9.6,
            "axes.titlesize": 11.4,
            "axes.labelsize": 10.1,
            "xtick.labelsize": 8.8,
            "ytick.labelsize": 8.8,
            "legend.fontsize": 8.6,
            "legend.frameon": False,
            "lines.linewidth": 1.8,
            "lines.markersize": 4.2,
            "path.simplify": False,
        }
    )
    if style_path is not None and style_path.exists():
        plt.style.use(style_path)


def finalize_figure(
    fig: mpl.figure.Figure,
    title: str,
    *,
    fontsize: float,
    top: float = 0.955,
    x: float = 0.02,
    ha: str = "left",
) -> None:
    """为总标题预留独立顶边距，避免与子图标题贴近。"""
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(rect=(0.0, 0.0, 1.0, top))
    else:
        fig.subplots_adjust(top=top)
    fig.suptitle(title, x=x, y=0.98, ha=ha, va="top", fontsize=fontsize)


def plot_runs(runs: list[RunMetrics], output_dir: Path) -> list[Path]:
    """绘制对比图并导出 PNG/PDF。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    colors = ["#A63D40", "#2F6690"]
    raw_colors = ["#E9C2C2", "#C8D8E8"]
    markers = ["o", "s"]
    fig, axes = plt.subplot_mosaic(
        [["loss", "loss"], ["top1", "lr"]],
        figsize=(7.1, 5.2),
        gridspec_kw={"height_ratios": [1.5, 1.0]},
        layout="constrained",
    )

    for idx, run in enumerate(runs):
        color = colors[idx % len(colors)]
        raw_color = raw_colors[idx % len(raw_colors)]
        marker = markers[idx % len(markers)]

        window = max(31, min(301, len(run.iter_loss) // 20))
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=float) / window
        smooth_loss = np.convolve(np.asarray(run.iter_loss), kernel, mode="same")

        axes["loss"].plot(
            run.global_steps, run.iter_loss,
            color=raw_color, linewidth=0.6, alpha=0.28, zorder=1
        )
        axes["loss"].plot(
            run.global_steps, smooth_loss,
            color=color, label=run.label, linewidth=1.55, zorder=3
        )
        mean_step_positions = np.linspace(run.global_steps[0], run.global_steps[-1], len(run.train_mean_loss))
        axes["loss"].plot(
            mean_step_positions, run.train_mean_loss,
            color=color, linewidth=0.0, marker=marker, markersize=3.2,
            markerfacecolor="white", markeredgewidth=0.9, zorder=4
        )

        best_index = int(np.argmax(run.eval_top1))
        best_epoch = run.eval_epochs[best_index]
        best_top1 = run.eval_top1[best_index]
        top1_label = (
            f"{run.label}\nbest {best_top1:.2f}, final {run.eval_top1[-1]:.2f}"
        )
        axes["top1"].plot(
            run.eval_epochs,
            run.eval_top1,
            color=color,
            label=top1_label,
            linewidth=1.7,
            marker=marker,
            markerfacecolor="white",
            markeredgewidth=1.2,
        )

        lr_step_positions = np.linspace(run.global_steps[0], run.global_steps[-1], len(run.epoch_lr))
        axes["lr"].step(
            lr_step_positions, run.epoch_lr,
            where="post", color=color, linewidth=2.0, alpha=0.9
        )
        for step in run.lr_steps:
            step_position = run.global_steps[-1] * ((step + 1) / max(run.total_epochs, 1))
            axes["lr"].axvline(
                step_position, color=color, alpha=0.2, linewidth=1.2, linestyle="--"
            )

    axes["loss"].set(title="Training Loss", xlabel="Global Step", ylabel="Cross-Entropy Loss")
    axes["top1"].set(title="Validation Top-1", xlabel="Epoch", ylabel="Accuracy (%)")
    axes["lr"].set(title="Learning Rate Schedule", xlabel="Global Step", ylabel="LR")
    axes["lr"].ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))

    max_global_step = max(run.global_steps[-1] for run in runs)
    axes["loss"].set_xlim(0, max_global_step)
    axes["lr"].set_xlim(0, max_global_step)
    max_epoch = max(run.eval_epochs[-1] for run in runs) + 2
    axes["top1"].set_xlim(1, max_epoch)
    axes["top1"].set_ylim(
        min(min(run.eval_top1) for run in runs) - 0.8,
        max(max(run.eval_top1) for run in runs) + 0.5,
    )
    axes["top1"].grid(axis="x", visible=False)
    axes["loss"].legend(loc="upper right", ncol=2, handlelength=2.0)
    axes["top1"].legend(loc="lower right", handlelength=1.5, fontsize=8.6)

    top1_delta = runs[1].eval_top1[-1] - runs[0].eval_top1[-1]
    finalize_figure(
        fig,
        f"NTU RGB+D xsub: paper LR schedule improves final Top-1 by {top1_delta:+.2f} points",
        fontsize=11.2,
        top=0.94,
        x=0.02,
        ha="left",
    )
    axes["loss"].text(
        0.01,
        0.96,
        "raw iter loss (thin) and moving average (bold)",
        transform=axes["loss"].transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        color="#5F6368",
    )

    png_path = output_dir / "ntu_xsub_training_comparison.png"
    pdf_path = output_dir / "ntu_xsub_training_comparison.pdf"
    fig.savefig(png_path)
    fig.savefig(pdf_path)
    plt.close(fig)
    return [png_path, pdf_path]


def build_default_runs() -> list[tuple[str, Path]]:
    """返回当前仓库的默认两组正式训练路径。"""
    return [
        (
            "Repo Default",
            ROOT_DIR / "work_dir" / "recognition" / "ntu-xsub" / "ST_GCN_fp32_bs16_lr0025_e80",
        ),
        (
            "Paper Schedule",
            ROOT_DIR
            / "work_dir"
            / "recognition"
            / "ntu-xsub"
            / "ST_GCN_fp32_bs16_lr0025_e80_step10x",
        ),
    ]


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    apply_style(args.style)

    run_specs = args.runs if args.runs is not None else build_default_runs()
    if len(run_specs) != 2:
        raise ValueError("当前脚本要求恰好提供两组运行结果。")

    runs = [parse_run(label, Path(work_dir)) for label, work_dir in run_specs]
    outputs = plot_runs(runs, args.output_dir)

    print("已导出图像：")
    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
