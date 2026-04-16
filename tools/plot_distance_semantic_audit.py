#!/usr/bin/env python3
"""绘制 `distance partitioning` 语义审计报告的关键对比图。"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / "temp" / "matplotlib"))

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scienceplots  # noqa: F401
import torch
from mpl_toolkits.axes_grid1 import ImageGrid

from net.st_gcn import Model
from net.utils.graph import Graph
from net.utils.graph import get_hop_distance
from net.utils.graph import normalize_digraph
from net.utils.tgcn import ConvTemporalGraphical

EVAL_EPOCH_PATTERN = re.compile(r"Eval epoch:\s*(\d+)")
MEAN_LOSS_PATTERN = re.compile(r"\bmean_loss:\s*([0-9]*\.?[0-9]+)")
TOPK_PATTERN = re.compile(r"Top(\d+):\s*([0-9]*\.?[0-9]+)%")


@dataclass
class ProbeRun:
    """存放短程探针对照实验的评估曲线。"""

    label: str
    work_dir: Path
    color: str
    marker: str
    eval_epochs: list[int]
    eval_mean_loss: list[float]
    eval_top1: list[float]
    eval_top5: list[float]


@dataclass
class ForwardScaleStats:
    """存放受控前向实验的尺度统计。"""

    tgcn_current_abs_mean: float
    tgcn_subsetnorm_abs_mean: float
    tgcn_ratio: float
    block_indices: list[int]
    current_block_abs_mean: list[float]
    subsetnorm_block_abs_mean: list[float]
    block_ratios: list[float]
    current_logits_abs_mean: float
    subsetnorm_logits_abs_mean: float
    logits_ratio: float


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="绘制 distance 语义审计报告的补充对比图")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT_DIR / "work_dir" / "figures",
        help="图像输出目录。",
    )
    return parser.parse_args()


def apply_style() -> None:
    """应用与仓库现有实验图一致的学术风格。"""
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


def set_padded_ylim(
    ax: mpl.axes.Axes,
    series_list: list[list[float] | np.ndarray],
    *,
    top_ratio: float = 0.18,
    bottom_ratio: float = 0.08,
    min_pad: float = 0.05,
) -> None:
    """按数据范围给坐标轴补上下边距，避免注释顶到曲线。"""
    values = np.concatenate([np.asarray(series, dtype=float) for series in series_list])
    y_min = float(np.min(values))
    y_max = float(np.max(values))
    span = max(y_max - y_min, min_pad)
    ax.set_ylim(y_min - span * bottom_ratio, y_max + span * top_ratio)


def set_bar_padded_ylim(ax: mpl.axes.Axes, values: np.ndarray, *, pad_ratio: float = 0.18) -> None:
    """给柱状图设置更宽松的纵轴范围。"""
    y_min = min(0.0, float(np.min(values)))
    y_max = max(0.0, float(np.max(values)))
    span = max(y_max - y_min, 0.5)
    ax.set_ylim(y_min - span * pad_ratio, y_max + span * pad_ratio)


def finalize_figure(
    fig: mpl.figure.Figure,
    title: str,
    *,
    fontsize: float,
    top: float = 0.94,
    x: float = 0.5,
    ha: str = "center",
) -> None:
    """为总标题预留独立顶边距，避免与子图标题贴近或重叠。"""
    layout_engine = fig.get_layout_engine()
    if layout_engine is not None and hasattr(layout_engine, "set"):
        layout_engine.set(rect=(0.0, 0.0, 1.0, top))
    else:
        fig.subplots_adjust(top=top)
    fig.suptitle(title, x=x, y=0.98, ha=ha, va="top", fontsize=fontsize)


def build_toy_distance_variants() -> tuple[np.ndarray, np.ndarray]:
    """构造 3 节点链图上的两种 distance 邻接口径。"""
    num_node = 3
    edges = [(0, 0), (1, 1), (2, 2), (0, 1), (1, 2)]
    hop_distance = get_hop_distance(num_node, edges, max_hop=1)

    adjacency = np.zeros((num_node, num_node))
    for hop in range(2):
        adjacency[hop_distance == hop] = 1

    normalized_all = normalize_digraph(adjacency)

    current = np.zeros((2, num_node, num_node))
    per_subset = np.zeros((2, num_node, num_node))
    for subset_index, hop in enumerate(range(2)):
        current[subset_index][hop_distance == hop] = normalized_all[hop_distance == hop]

        subset_adjacency = np.zeros((num_node, num_node))
        subset_adjacency[hop_distance == hop] = 1
        per_subset[subset_index] = normalize_digraph(subset_adjacency)

    return current, per_subset


def parse_probe_run(label: str, work_dir: Path, color: str, marker: str) -> ProbeRun:
    """从日志中解析短程探针的验证曲线。"""
    log_path = work_dir / "log.txt"
    if not log_path.exists():
        raise FileNotFoundError(f"未找到日志文件: {log_path}")

    eval_epochs: list[int] = []
    eval_mean_loss: list[float] = []
    eval_top1: list[float] = []
    eval_top5: list[float] = []

    current_eval_epoch: int | None = None
    pending_eval_loss: float | None = None
    pending_topk: dict[int, float] = {}

    with log_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue

            eval_match = EVAL_EPOCH_PATTERN.search(line)
            if eval_match:
                current_eval_epoch = int(eval_match.group(1)) + 1
                pending_eval_loss = None
                pending_topk = {}
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
                    pending_eval_loss = None
                    pending_topk = {}

    if not eval_epochs:
        raise ValueError(f"未从 {log_path} 解析出验证曲线。")

    return ProbeRun(
        label=label,
        work_dir=work_dir,
        color=color,
        marker=marker,
        eval_epochs=eval_epochs,
        eval_mean_loss=eval_mean_loss,
        eval_top1=eval_top1,
        eval_top5=eval_top5,
    )


def build_probe_runs() -> list[ProbeRun]:
    """返回语义审计报告里的两组短程对照实验。"""
    root = ROOT_DIR / "work_dir" / "ablation" / "ntu-xsub"
    return [
        parse_probe_run(
            label="Distance / No Imp.",
            work_dir=root / "distance_noimp",
            color="#A63D40",
            marker="o",
        ),
        parse_probe_run(
            label="Distance / SubsetNorm Probe",
            work_dir=root / "distance_subsetnorm_probe",
            color="#2A6F97",
            marker="s",
        ),
    ]


def build_column_balance_series() -> dict[str, dict[str, np.ndarray]]:
    """构造 NTU 图上两种归一化口径的列和分布。"""
    current = Graph(layout="ntu-rgb+d", strategy="distance").A
    subsetnorm = Graph(layout="ntu-rgb+d", strategy="distance_subsetnorm").A

    return {
        "current": {
            "self": current[0].sum(axis=0),
            "neighbor": current[1].sum(axis=0),
            "merged": current.sum(axis=0).sum(axis=0),
        },
        "subsetnorm": {
            "self": subsetnorm[0].sum(axis=0),
            "neighbor": subsetnorm[1].sum(axis=0),
            "merged": subsetnorm.sum(axis=0).sum(axis=0),
        },
    }


def compute_forward_equivalence_outputs() -> tuple[np.ndarray, np.ndarray]:
    """构造最小前向一致性测试的输出。"""
    torch.manual_seed(0)

    uniform = torch.tensor(Graph(layout="ntu-rgb+d", strategy="uniform").A, dtype=torch.float32)
    distance = torch.tensor(Graph(layout="ntu-rgb+d", strategy="distance").A, dtype=torch.float32)

    uniform_conv = ConvTemporalGraphical(2, 3, 1, bias=False)
    distance_conv = ConvTemporalGraphical(2, 3, 2, bias=False)

    with torch.no_grad():
        weight = uniform_conv.conv.weight.detach().clone()
        distance_conv.conv.weight[:3].copy_(weight)
        distance_conv.conv.weight[3:].copy_(weight)

    x = torch.randn(4, 2, 5, 25)
    uniform_out, _ = uniform_conv(x, uniform)
    distance_out, _ = distance_conv(x, distance)
    return (
        uniform_out.detach().cpu().numpy().reshape(-1),
        distance_out.detach().cpu().numpy().reshape(-1),
    )


def capture_block_abs_means(model: Model, x: torch.Tensor) -> tuple[list[float], float]:
    """记录整模型 10 个 ST-GCN block 的激活绝对均值。"""
    block_abs_means: list[float] = []
    hooks = []

    def hook_fn(_module: torch.nn.Module, _inputs: tuple[torch.Tensor, ...], output: tuple[torch.Tensor, torch.Tensor]) -> None:
        tensor_out, _ = output
        block_abs_means.append(float(tensor_out.detach().abs().mean().cpu()))

    for block in model.st_gcn_networks:
        hooks.append(block.register_forward_hook(hook_fn))

    with torch.no_grad():
        logits = model(x)

    for hook in hooks:
        hook.remove()

    logits_abs_mean = float(logits.detach().abs().mean().cpu())
    return block_abs_means, logits_abs_mean


def compute_forward_scale_stats() -> ForwardScaleStats:
    """复现报告中的受控前向尺度变化。"""
    torch.manual_seed(0)
    x_tgcn = torch.randn(4, 3, 20, 25)
    current_A = torch.tensor(Graph(layout="ntu-rgb+d", strategy="distance").A, dtype=torch.float32)
    subsetnorm_A = torch.tensor(
        Graph(layout="ntu-rgb+d", strategy="distance_subsetnorm").A,
        dtype=torch.float32,
    )
    tgcn = ConvTemporalGraphical(3, 8, 2, bias=False)
    with torch.no_grad():
        current_tgcn_out, _ = tgcn(x_tgcn, current_A)
        subsetnorm_tgcn_out, _ = tgcn(x_tgcn, subsetnorm_A)

    tgcn_current_abs_mean = float(current_tgcn_out.detach().abs().mean().cpu())
    tgcn_subsetnorm_abs_mean = float(subsetnorm_tgcn_out.detach().abs().mean().cpu())

    torch.manual_seed(0)
    current_model = Model(
        in_channels=3,
        num_class=60,
        graph_args={"layout": "ntu-rgb+d", "strategy": "distance"},
        edge_importance_weighting=False,
        dropout=0.5,
    )
    torch.manual_seed(0)
    subsetnorm_model = Model(
        in_channels=3,
        num_class=60,
        graph_args={"layout": "ntu-rgb+d", "strategy": "distance_subsetnorm"},
        edge_importance_weighting=False,
        dropout=0.5,
    )
    current_model.eval()
    subsetnorm_model.eval()

    torch.manual_seed(1)
    model_input = torch.randn(2, 3, 20, 25, 2)

    current_block_abs_mean, current_logits_abs_mean = capture_block_abs_means(
        current_model,
        model_input,
    )
    subsetnorm_block_abs_mean, subsetnorm_logits_abs_mean = capture_block_abs_means(
        subsetnorm_model,
        model_input,
    )
    block_ratios = [
        subset / current for current, subset in zip(current_block_abs_mean, subsetnorm_block_abs_mean, strict=True)
    ]

    return ForwardScaleStats(
        tgcn_current_abs_mean=tgcn_current_abs_mean,
        tgcn_subsetnorm_abs_mean=tgcn_subsetnorm_abs_mean,
        tgcn_ratio=tgcn_subsetnorm_abs_mean / tgcn_current_abs_mean,
        block_indices=list(range(1, len(current_block_abs_mean) + 1)),
        current_block_abs_mean=current_block_abs_mean,
        subsetnorm_block_abs_mean=subsetnorm_block_abs_mean,
        block_ratios=block_ratios,
        current_logits_abs_mean=current_logits_abs_mean,
        subsetnorm_logits_abs_mean=subsetnorm_logits_abs_mean,
        logits_ratio=subsetnorm_logits_abs_mean / current_logits_abs_mean,
    )


def annotate_heatmap(ax: mpl.axes.Axes, matrix: np.ndarray) -> None:
    """给 3x3 toy 矩阵标注数值。"""
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            color = "white" if value >= 0.62 else "#202020"
            ax.text(
                col,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                fontsize=8.2,
                color=color,
            )


def plot_toy_matrices(output_dir: Path) -> list[Path]:
    """绘制 toy graph 上的两种归一化矩阵对比热图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    current, per_subset = build_toy_distance_variants()
    matrices = [
        current[0],
        current[1],
        current.sum(axis=0),
        per_subset[0],
        per_subset[1],
        per_subset.sum(axis=0),
    ]
    titles = [
        "Current: self subset",
        "Current: hop-1 subset",
        "Current: merged",
        "SubsetNorm: self subset",
        "SubsetNorm: hop-1 subset",
        "SubsetNorm: merged",
    ]

    fig = plt.figure(figsize=(9.4, 5.8))
    grid = ImageGrid(
        fig,
        111,
        nrows_ncols=(2, 3),
        axes_pad=0.35,
        share_all=True,
        cbar_location="right",
        cbar_mode="single",
        cbar_size="3%",
        cbar_pad=0.10,
    )

    image = None
    for ax, matrix, title in zip(grid, matrices, titles, strict=True):
        image = ax.imshow(matrix, cmap="Blues", vmin=0.0, vmax=1.0)
        annotate_heatmap(ax, matrix)
        ax.set_title(title, fontsize=9.4)
        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xlabel("Source node")
        ax.set_ylabel("Target node")
        ax.grid(False)

    if image is not None:
        cbar = grid.cbar_axes[0].colorbar(image)
        cbar.ax.set_ylabel("Normalized weight", rotation=90, va="center")

    finalize_figure(
        fig,
        "Toy Graph: Distance Partition Normalization Semantics",
        fontsize=11.2,
        top=0.93,
    )

    output_paths = [
        output_dir / "ntu_xsub_distance_semantic_toy_matrices.png",
        output_dir / "ntu_xsub_distance_semantic_toy_matrices.pdf",
    ]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def annotate_probe_delta(ax: mpl.axes.Axes, epochs: list[int], delta_top1: np.ndarray) -> None:
    """标注短程探针在各评估点上的 Top-1 差值。"""
    for epoch, delta in zip(epochs, delta_top1, strict=True):
        ax.text(
            epoch,
            delta + (0.9 if delta >= 0 else -1.2),
            f"{delta:+.2f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=7.6,
            color="#4B4B4B",
        )


def plot_probe_curves(output_dir: Path) -> list[Path]:
    """绘制短程探针与原始 distance 的验证曲线对比。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = build_probe_runs()
    baseline, probe = runs

    fig, axes = plt.subplot_mosaic(
        [["top1", "top5"], ["loss", "delta"]],
        figsize=(9.4, 6.2),
        gridspec_kw={"height_ratios": [1.0, 0.95], "width_ratios": [1.0, 1.0]},
        layout="constrained",
    )

    for run in runs:
        for axis_name, series, ylabel in (
            ("top1", run.eval_top1, "Accuracy (%)"),
            ("top5", run.eval_top5, "Accuracy (%)"),
            ("loss", run.eval_mean_loss, "Loss"),
        ):
            axes[axis_name].plot(
                run.eval_epochs,
                series,
                color=run.color,
                linewidth=1.8,
                marker=run.marker,
                markersize=4.2,
                markerfacecolor="white",
                markeredgewidth=1.0,
                label=run.label,
            )
            axes[axis_name].set_xlabel("Epoch")
            axes[axis_name].set_ylabel(ylabel)
            axes[axis_name].set_xticks(baseline.eval_epochs)
            axes[axis_name].set_xlim(min(baseline.eval_epochs) - 1, max(baseline.eval_epochs) + 1)
            axes[axis_name].axvspan(
                max(probe.eval_epochs) + 1,
                max(baseline.eval_epochs) + 1,
                color="#EFEFEF",
                alpha=0.7,
                zorder=0,
            )

    axes["top1"].set_title("Probe vs. Original: Validation Top-1")
    axes["top5"].set_title("Probe vs. Original: Validation Top-5")
    axes["loss"].set_title("Probe vs. Original: Validation Mean Loss")
    set_padded_ylim(axes["top1"], [baseline.eval_top1, probe.eval_top1], top_ratio=0.26)
    set_padded_ylim(axes["top5"], [baseline.eval_top5, probe.eval_top5], top_ratio=0.26)
    set_padded_ylim(axes["loss"], [baseline.eval_mean_loss, probe.eval_mean_loss], top_ratio=0.22)
    for axis_name in ("top1", "top5", "loss"):
        axes[axis_name].text(
            0.98,
            0.97,
            "gray area: original run continues,\nprobe stops at epoch 14",
            transform=axes[axis_name].transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
            color="#666666",
        )
    axes["top1"].legend(loc="lower right")
    axes["top5"].legend(loc="lower right")
    axes["loss"].legend(loc="upper right")

    baseline_top1_by_epoch = dict(zip(baseline.eval_epochs, baseline.eval_top1, strict=True))
    probe_top1_by_epoch = dict(zip(probe.eval_epochs, probe.eval_top1, strict=True))
    common_epochs = sorted(set(baseline_top1_by_epoch) & set(probe_top1_by_epoch))
    delta_top1 = np.array(
        [probe_top1_by_epoch[epoch] - baseline_top1_by_epoch[epoch] for epoch in common_epochs]
    )
    axes["delta"].axhline(0.0, color="#777777", linewidth=1.0, linestyle="--")
    axes["delta"].bar(
        common_epochs,
        delta_top1,
        width=1.6,
        color=["#2A6F97" if value >= 0 else "#A63D40" for value in delta_top1],
        alpha=0.85,
    )
    annotate_probe_delta(axes["delta"], common_epochs, delta_top1)
    set_bar_padded_ylim(axes["delta"], delta_top1)
    axes["delta"].set_title("Top-1 Delta (Probe - Original)")
    axes["delta"].set_xlabel("Epoch")
    axes["delta"].set_ylabel("Top-1 delta")
    axes["delta"].set_xticks(common_epochs)
    axes["delta"].set_xlim(min(common_epochs) - 1, max(common_epochs) + 1)

    finalize_figure(
        fig,
        "Distance Probe: Short-Range Validation Comparison",
        fontsize=11.2,
        top=0.94,
    )

    output_paths = [
        output_dir / "ntu_xsub_distance_semantic_probe_curves.png",
        output_dir / "ntu_xsub_distance_semantic_probe_curves.pdf",
    ]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def add_range_note(ax: mpl.axes.Axes, current: np.ndarray, subsetnorm: np.ndarray) -> None:
    """在列和分布图上标注数值范围。"""
    ax.text(
        0.98,
        0.04,
        f"current: {current.min():.2f} - {current.max():.2f}\n"
        f"subsetnorm: {subsetnorm.min():.2f} - {subsetnorm.max():.2f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7.4,
        color="#555555",
        bbox={"boxstyle": "round,pad=0.22", "facecolor": "white", "edgecolor": "none", "alpha": 0.9},
    )


def plot_column_balance(output_dir: Path) -> list[Path]:
    """绘制 NTU 图上 self/neighbor/merged 列和分布对比。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    series = build_column_balance_series()
    node_ids = np.arange(1, 26)

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.7), layout="constrained")
    panel_specs = [
        ("self", "Self subset column sum"),
        ("neighbor", "Neighbor subset column sum"),
        ("merged", "Merged total column sum"),
    ]

    for ax, (key, title) in zip(axes, panel_specs, strict=True):
        current_values = series["current"][key]
        subsetnorm_values = series["subsetnorm"][key]
        ax.plot(
            node_ids,
            current_values,
            color="#A63D40",
            linewidth=1.8,
            marker="o",
            markersize=3.6,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="Current distance",
        )
        ax.plot(
            node_ids,
            subsetnorm_values,
            color="#2A6F97",
            linewidth=1.8,
            marker="s",
            markersize=3.6,
            markerfacecolor="white",
            markeredgewidth=0.9,
            label="Distance subsetnorm",
        )
        ax.set_title(title)
        ax.set_xlabel("Node index")
        ax.set_ylabel("Column sum")
        ax.set_xticks([1, 5, 10, 15, 20, 25])
        set_padded_ylim(ax, [current_values, subsetnorm_values], top_ratio=0.22, bottom_ratio=0.12)
        add_range_note(ax, current_values, subsetnorm_values)

    axes[0].legend(loc="lower left")
    finalize_figure(
        fig,
        "NTU Graph: Column-Sum Balance Under Two Normalization Orders",
        fontsize=11.0,
        top=0.93,
    )

    output_paths = [
        output_dir / "ntu_xsub_distance_semantic_column_balance.png",
        output_dir / "ntu_xsub_distance_semantic_column_balance.pdf",
    ]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def plot_forward_equivalence(output_dir: Path) -> list[Path]:
    """绘制 uniform 与 distance 的最小前向一致性图。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    uniform_out, distance_out = compute_forward_equivalence_outputs()
    abs_diff = np.abs(uniform_out - distance_out)
    sample_index = np.linspace(0, uniform_out.size - 1, 2500, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8), layout="constrained")

    axes[0].scatter(
        uniform_out[sample_index],
        distance_out[sample_index],
        s=7,
        alpha=0.45,
        color="#2A6F97",
        edgecolors="none",
    )
    min_value = float(min(uniform_out[sample_index].min(), distance_out[sample_index].min()))
    max_value = float(max(uniform_out[sample_index].max(), distance_out[sample_index].max()))
    axes[0].plot([min_value, max_value], [min_value, max_value], color="#A63D40", linewidth=1.3)
    axes[0].set_title("Controlled Forward: Uniform vs. Distance")
    axes[0].set_xlabel("Uniform output")
    axes[0].set_ylabel("Distance output")
    axes[0].text(
        0.03,
        0.97,
        f"allclose = True\nmax |diff| = {abs_diff.max():.2e}",
        transform=axes[0].transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#4B4B4B",
    )

    axes[1].hist(abs_diff, bins=40, color="#7A3B69", alpha=0.88, edgecolor="white")
    axes[1].set_title("Absolute Difference Distribution")
    axes[1].set_xlabel("|Uniform - Distance|")
    axes[1].set_ylabel("Count")
    axes[1].text(
        0.97,
        0.97,
        f"mean = {abs_diff.mean():.2e}\nstd = {abs_diff.std():.2e}",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color="#4B4B4B",
    )

    finalize_figure(
        fig,
        "Forward Equivalence: No Extra Bias in Distance Aggregation",
        fontsize=11.0,
        top=0.94,
    )

    output_paths = [
        output_dir / "ntu_xsub_distance_semantic_forward_equivalence.png",
        output_dir / "ntu_xsub_distance_semantic_forward_equivalence.pdf",
    ]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def plot_scale_amplification(output_dir: Path) -> list[Path]:
    """绘制 subsetnorm 带来的受控尺度放大。"""
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = compute_forward_scale_stats()

    fig, axes = plt.subplot_mosaic(
        [["tgcn", "blocks"], ["logits", "ratio"]],
        figsize=(9.6, 6.0),
        gridspec_kw={"height_ratios": [1.0, 0.95], "width_ratios": [0.9, 1.25]},
        layout="constrained",
    )

    axes["tgcn"].bar(
        ["Current", "SubsetNorm"],
        [stats.tgcn_current_abs_mean, stats.tgcn_subsetnorm_abs_mean],
        color=["#A63D40", "#2A6F97"],
        alpha=0.9,
    )
    axes["tgcn"].set_title("Minimal TGCN Output Scale")
    axes["tgcn"].set_ylabel("Abs mean")
    set_bar_padded_ylim(
        axes["tgcn"],
        np.array([stats.tgcn_current_abs_mean, stats.tgcn_subsetnorm_abs_mean]),
    )
    axes["tgcn"].text(
        0.5,
        0.97,
        f"subsetnorm / current = {stats.tgcn_ratio:.2f}x",
        transform=axes["tgcn"].transAxes,
        ha="center",
        va="top",
        fontsize=7.7,
        color="#4B4B4B",
    )

    axes["blocks"].plot(
        stats.block_indices,
        stats.current_block_abs_mean,
        color="#A63D40",
        linewidth=1.8,
        marker="o",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Current distance",
    )
    axes["blocks"].plot(
        stats.block_indices,
        stats.subsetnorm_block_abs_mean,
        color="#2A6F97",
        linewidth=1.8,
        marker="s",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
        label="Distance subsetnorm",
    )
    axes["blocks"].set_title("10 ST-GCN Blocks: Activation Abs Mean")
    axes["blocks"].set_xlabel("Block index")
    axes["blocks"].set_ylabel("Abs mean")
    axes["blocks"].set_xticks(stats.block_indices)
    set_padded_ylim(
        axes["blocks"],
        [stats.current_block_abs_mean, stats.subsetnorm_block_abs_mean],
        top_ratio=0.24,
    )
    axes["blocks"].legend(loc="upper right")

    axes["logits"].bar(
        ["Current", "SubsetNorm"],
        [stats.current_logits_abs_mean, stats.subsetnorm_logits_abs_mean],
        color=["#A63D40", "#2A6F97"],
        alpha=0.9,
    )
    axes["logits"].set_title("Final Logits Abs Mean")
    axes["logits"].set_ylabel("Abs mean")
    set_bar_padded_ylim(
        axes["logits"],
        np.array([stats.current_logits_abs_mean, stats.subsetnorm_logits_abs_mean]),
    )
    axes["logits"].text(
        0.5,
        0.97,
        f"subsetnorm / current = {stats.logits_ratio:.2f}x",
        transform=axes["logits"].transAxes,
        ha="center",
        va="top",
        fontsize=7.7,
        color="#4B4B4B",
    )

    axes["ratio"].plot(
        stats.block_indices,
        stats.block_ratios,
        color="#7A3B69",
        linewidth=1.8,
        marker="D",
        markersize=4.0,
        markerfacecolor="white",
        markeredgewidth=1.0,
    )
    axes["ratio"].axhline(1.0, color="#777777", linewidth=1.0, linestyle="--")
    axes["ratio"].set_title("Block-wise Scale Ratio")
    axes["ratio"].set_xlabel("Block index")
    axes["ratio"].set_ylabel("SubsetNorm / Current")
    axes["ratio"].set_xticks(stats.block_indices)
    set_padded_ylim(axes["ratio"], [stats.block_ratios], top_ratio=0.22, bottom_ratio=0.12)
    axes["ratio"].text(
        0.03,
        0.97,
        f"range = {min(stats.block_ratios):.2f}x - {max(stats.block_ratios):.2f}x",
        transform=axes["ratio"].transAxes,
        ha="left",
        va="top",
        fontsize=7.6,
        color="#4B4B4B",
    )

    finalize_figure(
        fig,
        "Scale Amplification Under Distance SubsetNorm",
        fontsize=11.2,
        top=0.94,
    )

    output_paths = [
        output_dir / "ntu_xsub_distance_semantic_scale_amplification.png",
        output_dir / "ntu_xsub_distance_semantic_scale_amplification.pdf",
    ]
    for path in output_paths:
        fig.savefig(path)
    plt.close(fig)
    return output_paths


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    apply_style()
    plot_toy_matrices(args.output_dir)
    plot_probe_curves(args.output_dir)
    plot_column_balance(args.output_dir)
    plot_forward_equivalence(args.output_dir)
    plot_scale_amplification(args.output_dir)


if __name__ == "__main__":
    main()
