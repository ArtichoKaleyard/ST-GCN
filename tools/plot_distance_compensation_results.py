#!/usr/bin/env python3
"""绘制 ST-GCN distance partitioning 补偿假设实验结果与权重统计。"""

from __future__ import annotations

import argparse
import csv
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
EVAL_EPOCH_PATTERN = re.compile(r"Eval epoch:\s*(\d+)")
MEAN_LOSS_PATTERN = re.compile(r"\bmean_loss:\s*([0-9]*\.?[0-9]+)")
TOPK_PATTERN = re.compile(r"Top(\d+):\s*([0-9]*\.?[0-9]+)%")


@dataclass(frozen=True)
class RunSpec:
    """描述一组实验及其绘图属性。"""

    key: str
    label: str
    work_dir: Path
    color: str
    linestyle: str
    marker: str
    stage: str


@dataclass
class ExperimentRun:
    """存放单组实验的训练与验证曲线。"""

    key: str
    label: str
    work_dir: Path
    color: str
    linestyle: str
    marker: str
    stage: str
    total_epochs: int
    eval_epochs: list[int]
    eval_mean_loss: list[float]
    eval_top1: list[float]
    eval_top5: list[float]

    @property
    def final_top1(self) -> float:
        """返回最终 Top-1。"""
        return self.eval_top1[-1]

    @property
    def final_top5(self) -> float:
        """返回最终 Top-5。"""
        return self.eval_top5[-1]

    @property
    def best_top1(self) -> float:
        """返回最佳 Top-1。"""
        return max(self.eval_top1)

    @property
    def best_epoch(self) -> int:
        """返回最佳 Top-1 对应 epoch。"""
        return self.eval_epochs[int(np.argmax(self.eval_top1))]


@dataclass
class WeightSeries:
    """存放单个模型的有效邻接统计曲线。"""

    label: str
    ratio_label: str
    layers: list[int]
    self_mask_mean: list[float]
    neighbor_mask_mean: list[float]
    self_colsum_mean: list[float]
    neighbor_colsum_mean: list[float]
    self_over_neighbor_mask: list[float]
    self_over_neighbor_colsum: list[float]


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="绘制 distance compensation 实验结果")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "work_dir" / "figures",
        help="图像输出目录。",
    )
    return parser.parse_args()


def apply_style() -> None:
    """应用与既有消融图一致的学术风格。"""
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
            "font.size": 9.2,
            "axes.titlesize": 10.6,
            "axes.labelsize": 9.8,
            "xtick.labelsize": 8.3,
            "ytick.labelsize": 8.3,
            "legend.fontsize": 8.0,
            "legend.frameon": False,
        }
    )


def load_yaml(path: Path) -> dict[str, Any]:
    """读取 YAML 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_run_specs() -> list[RunSpec]:
    """返回六组实验的绘图规范。"""
    root = ROOT_DIR / "work_dir" / "ablation" / "ntu-xsub"
    return [
        RunSpec(
            key="distance_imp_20",
            label="Dist. + Imp. (20e)",
            work_dir=root / "distance_imp",
            color="#A63D40",
            linestyle="-",
            marker="o",
            stage="20 epoch",
        ),
        RunSpec(
            key="distance_rescaled_noimp_20",
            label="Dist. + Rescaled (20e)",
            work_dir=root / "distance_subsetnorm_rescaled_noimp",
            color="#E09F3E",
            linestyle="-",
            marker="s",
            stage="20 epoch",
        ),
        RunSpec(
            key="distance_rescaled_imp_20",
            label="Dist. + Rescaled + Imp. (20e)",
            work_dir=root / "distance_subsetnorm_rescaled_imp",
            color="#2A6F97",
            linestyle="-",
            marker="^",
            stage="20 epoch",
        ),
        RunSpec(
            key="spatial_rescaled_noimp_20",
            label="Spatial + Rescaled (20e)",
            work_dir=root / "spatial_subsetnorm_rescaled_noimp",
            color="#3A7D44",
            linestyle="-",
            marker="D",
            stage="20 epoch",
        ),
        RunSpec(
            key="distance_imp_40",
            label="Dist. + Imp. (40e)",
            work_dir=root / "distance_imp_e40",
            color="#A63D40",
            linestyle="--",
            marker="o",
            stage="40 epoch",
        ),
        RunSpec(
            key="distance_rescaled_imp_40",
            label="Dist. + Rescaled + Imp. (40e)",
            work_dir=root / "distance_subsetnorm_rescaled_imp_e40",
            color="#2A6F97",
            linestyle="--",
            marker="^",
            stage="40 epoch",
        ),
    ]


def parse_run(spec: RunSpec) -> ExperimentRun:
    """解析单组实验日志。

    Args:
        spec: 实验规格。

    Returns:
        解析后的实验曲线。

    Raises:
        FileNotFoundError: 配置或日志缺失。
        ValueError: 关键曲线未解析出来。
    """
    config_path = spec.work_dir / "config.yaml"
    log_path = spec.work_dir / "log.txt"
    if not config_path.exists():
        raise FileNotFoundError(f"未找到配置文件: {config_path}")
    if not log_path.exists():
        raise FileNotFoundError(f"未找到日志文件: {log_path}")

    config = load_yaml(config_path)
    eval_epochs: list[int] = []
    eval_mean_loss: list[float] = []
    eval_top1: list[float] = []
    eval_top5: list[float] = []

    current_eval_epoch: int | None = None
    pending_topk: dict[int, float] = {}
    pending_eval_loss: float | None = None

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            if TRAIN_EPOCH_PATTERN.search(line):
                continue

            eval_match = EVAL_EPOCH_PATTERN.search(line)
            if eval_match:
                current_eval_epoch = int(eval_match.group(1)) + 1
                pending_topk = {}
                pending_eval_loss = None
                continue

            if current_eval_epoch is None:
                continue

            mean_loss_match = MEAN_LOSS_PATTERN.search(line)
            if mean_loss_match and pending_eval_loss is None:
                pending_eval_loss = float(mean_loss_match.group(1))
                continue

            topk_match = TOPK_PATTERN.search(line)
            if topk_match:
                pending_topk[int(topk_match.group(1))] = float(topk_match.group(2))
                if 1 in pending_topk and 5 in pending_topk and pending_eval_loss is not None:
                    eval_epochs.append(current_eval_epoch)
                    eval_mean_loss.append(pending_eval_loss)
                    eval_top1.append(pending_topk[1])
                    eval_top5.append(pending_topk[5])
                    current_eval_epoch = None
                    pending_topk = {}
                    pending_eval_loss = None

    if not eval_epochs:
        raise ValueError(f"未从 {log_path} 解析出评估曲线。")

    return ExperimentRun(
        key=spec.key,
        label=spec.label,
        work_dir=spec.work_dir,
        color=spec.color,
        linestyle=spec.linestyle,
        marker=spec.marker,
        stage=spec.stage,
        total_epochs=int(config.get("num_epoch", 0)),
        eval_epochs=eval_epochs,
        eval_mean_loss=eval_mean_loss,
        eval_top1=eval_top1,
        eval_top5=eval_top5,
    )


def parse_weight_stats(label: str, ratio_label: str, csv_path: Path) -> WeightSeries:
    """读取 `A_eff` 统计 CSV。

    Args:
        label: 图中模型标签。
        ratio_label: 比值图中的模型标签。
        csv_path: 统计 CSV 路径。

    Returns:
        整理后的权重统计。
    """
    rows: dict[int, dict[str, dict[str, float]]] = {}
    with csv_path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            layer = int(row["layer"])
            subset_name = row["subset_name"]
            rows.setdefault(layer, {})[subset_name] = {
                "mask_mean": float(row["mask_mean"]),
                "a_eff_colsum_mean": float(row["a_eff_colsum_mean"]),
            }

    layers = sorted(rows)
    return WeightSeries(
        label=label,
        ratio_label=ratio_label,
        layers=layers,
        self_mask_mean=[rows[layer]["self"]["mask_mean"] for layer in layers],
        neighbor_mask_mean=[rows[layer]["neighbor"]["mask_mean"] for layer in layers],
        self_colsum_mean=[rows[layer]["self"]["a_eff_colsum_mean"] for layer in layers],
        neighbor_colsum_mean=[rows[layer]["neighbor"]["a_eff_colsum_mean"] for layer in layers],
        self_over_neighbor_mask=[rows[layer]["self_over_neighbor"]["mask_mean"] for layer in layers],
        self_over_neighbor_colsum=[
            rows[layer]["self_over_neighbor"]["a_eff_colsum_mean"] for layer in layers
        ],
    )


def add_stage_guides(ax: mpl.axes.Axes) -> None:
    """添加阶段说明。"""
    ax.text(
        0.02,
        0.98,
        "solid = 20 epoch, dashed = 40 epoch",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#666666",
    )


def annotate_summary(ax: mpl.axes.Axes, runs: list[ExperimentRun], positions: np.ndarray) -> None:
    """为摘要条形图添加文字。"""
    for pos, run in zip(positions, runs):
        ax.text(
            run.best_top1 + 0.18,
            pos + 0.14,
            f"{run.best_top1:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#4B4B4B",
        )
        ax.text(
            run.final_top1 + 0.18,
            pos - 0.14,
            f"{run.final_top1:.2f}",
            va="center",
            ha="left",
            fontsize=7.5,
            color="#4B4B4B",
        )
        ax.text(
            0.02,
            pos + 0.31,
            f"best epoch {run.best_epoch}",
            transform=ax.get_yaxis_transform(),
            va="bottom",
            ha="left",
            fontsize=7.2,
            color="#777777",
        )


def plot_experiment_results(runs: list[ExperimentRun], output_dir: Path) -> list[Path]:
    """绘制六组实验结果总图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplot_mosaic(
        [["top1", "top5"], ["valloss", "summary"]],
        figsize=(10.4, 6.4),
        gridspec_kw={"height_ratios": [1.0, 1.0], "width_ratios": [1.0, 1.0]},
        constrained_layout=True,
    )

    ordered_runs = runs
    y_positions = np.arange(len(ordered_runs))

    for run in runs:
        for axis_name, series in (
            ("top1", run.eval_top1),
            ("top5", run.eval_top5),
            ("valloss", run.eval_mean_loss),
        ):
            axes[axis_name].plot(
                run.eval_epochs,
                series,
                color=run.color,
                linewidth=1.55 if run.stage == "20 epoch" else 1.7,
                linestyle=run.linestyle,
                marker=run.marker,
                markersize=4.0,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=run.label,
            )

    for axis_name, title, ylabel in (
        ("top1", "Validation Top-1", "Accuracy (%)"),
        ("top5", "Validation Top-5", "Accuracy (%)"),
        ("valloss", "Validation Mean Loss", "Loss"),
    ):
        axes[axis_name].set_title(title)
        axes[axis_name].set_xlabel("Epoch")
        axes[axis_name].set_ylabel(ylabel)
        axes[axis_name].set_xticks([5, 10, 15, 20, 25, 30, 35, 40])
        axes[axis_name].set_xlim(3.5, 41.5)
        add_stage_guides(axes[axis_name])

    axes["top1"].legend(loc="lower right", ncol=1)
    axes["top5"].legend(loc="lower right", ncol=1)
    axes["valloss"].legend(loc="upper right", ncol=1)

    axes["summary"].barh(
        y_positions + 0.14,
        [run.best_top1 for run in ordered_runs],
        height=0.24,
        color="#D9D9D9",
        edgecolor="none",
    )
    axes["summary"].barh(
        y_positions - 0.14,
        [run.final_top1 for run in ordered_runs],
        height=0.24,
        color=[run.color for run in ordered_runs],
        alpha=1.0,
        edgecolor=[run.color for run in ordered_runs],
        linewidth=0.8,
    )
    annotate_summary(axes["summary"], ordered_runs, y_positions)
    axes["summary"].set_yticks(y_positions, [run.label for run in ordered_runs])
    axes["summary"].invert_yaxis()
    axes["summary"].set_xlabel("Top-1 (%)")
    axes["summary"].set_title("Best / Final Top-1")
    summary_values = [run.best_top1 for run in ordered_runs] + [run.final_top1 for run in ordered_runs]
    summary_min = min(summary_values)
    summary_max = max(summary_values)
    left_bound = max(0.0, np.floor((summary_min - 1.5) * 2.0) / 2.0)
    right_bound = np.ceil((summary_max + 0.8) * 2.0) / 2.0
    axes["summary"].set_xlim(left_bound, right_bound)
    axes["summary"].text(
        0.98,
        0.98,
        "gray bars = best, colored bars = final",
        transform=axes["summary"].transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#666666",
    )

    fig.suptitle("ST-GCN Distance Compensation Experiments", y=1.02, fontsize=11.6)

    output_paths = [
        output_dir / "ntu_xsub_distance_compensation_results.png",
        output_dir / "ntu_xsub_distance_compensation_results.pdf",
    ]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def plot_weight_analysis(weight_runs: list[WeightSeries], output_dir: Path) -> list[Path]:
    """绘制有效邻接权重分析图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(9.1, 6.5), constrained_layout=True)

    subset_colors = {"self": "#A63D40", "neighbor": "#2A6F97"}
    ratio_colors = ["#7A3B69", "#206A5D"]

    for idx, run in enumerate(weight_runs):
        top_ax = axes[0, idx]
        bottom_ax = axes[1, idx]
        top_ax.plot(
            run.layers,
            run.self_mask_mean,
            color=subset_colors["self"],
            linewidth=1.8,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="self mask",
        )
        top_ax.plot(
            run.layers,
            run.neighbor_mask_mean,
            color=subset_colors["neighbor"],
            linewidth=1.8,
            marker="s",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="neighbor mask",
        )
        top_ax.set_title(f"{run.label}: Edge Importance")
        top_ax.set_xlabel("Layer")
        top_ax.set_ylabel("Mask Mean")
        top_ax.set_xticks(run.layers)
        top_ax.set_ylim(0.47, 0.53)
        top_ax.legend(loc="lower right")
        top_ax.text(
            0.02,
            0.97,
            f"mean self/neighbor mask ratio = "
            f"{np.mean(run.self_over_neighbor_mask):.3f}",
            transform=top_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color="#666666",
        )

        bottom_ax.plot(
            run.layers,
            run.self_colsum_mean,
            color=subset_colors["self"],
            linewidth=1.9,
            marker="o",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="self $A_{eff}$",
        )
        bottom_ax.plot(
            run.layers,
            run.neighbor_colsum_mean,
            color=subset_colors["neighbor"],
            linewidth=1.9,
            marker="s",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=1.0,
            label="neighbor $A_{eff}$",
        )
        bottom_ax.plot(
            run.layers,
            run.self_over_neighbor_colsum,
            color=ratio_colors[idx],
            linewidth=1.7,
            linestyle="--",
            marker="^",
            markersize=4.0,
            markerfacecolor="white",
            markeredgewidth=0.95,
            label="self / neighbor ratio",
        )
        bottom_ax.axhline(1.0, color="#888888", linewidth=1.0, linestyle=":")
        bottom_ax.set_title(f"{run.label}: Effective Adjacency")
        bottom_ax.set_xlabel("Layer")
        bottom_ax.set_ylabel("Strength")
        bottom_ax.set_xticks(run.layers)
        bottom_ax.legend(loc="upper right")
        bottom_ax.text(
            0.02,
            0.97,
            f"mean self/neighbor colsum ratio = "
            f"{np.mean(run.self_over_neighbor_colsum):.3f}",
            transform=bottom_ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color="#666666",
        )

    fig.suptitle("Effective Adjacency Analysis for Distance + Importance", y=1.02, fontsize=11.6)
    output_paths = [
        output_dir / "ntu_xsub_distance_compensation_weights.png",
        output_dir / "ntu_xsub_distance_compensation_weights.pdf",
    ]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    apply_style()

    run_specs = build_run_specs()
    runs = [parse_run(spec) for spec in run_specs]
    result_paths = plot_experiment_results(runs, args.output_dir)

    root = ROOT_DIR / "work_dir" / "ablation" / "ntu-xsub"
    weight_runs = [
        parse_weight_stats(
            label="Distance + Imp. (40e)",
            ratio_label="Distance + Imp. (40e)",
            csv_path=root / "distance_imp_e40" / "stats" / "effective_adjacency.csv",
        ),
        parse_weight_stats(
            label="Distance + Rescaled + Imp. (40e)",
            ratio_label="Distance + Rescaled + Imp. (40e)",
            csv_path=root
            / "distance_subsetnorm_rescaled_imp_e40"
            / "stats"
            / "effective_adjacency.csv",
        ),
    ]
    weight_paths = plot_weight_analysis(weight_runs, args.output_dir)

    for path in [*result_paths, *weight_paths]:
        print(path)


if __name__ == "__main__":
    main()
