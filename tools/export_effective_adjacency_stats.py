"""导出 ST-GCN 有效邻接的轻量统计。

该脚本用于第二轮 `distance partitioning` 补偿假设实验。它不会绘制热图，
只会读取训练后的 checkpoint，恢复模型中的 `edge_importance`，并对每一层
的有效邻接 `A_eff = A ⊙ M` 输出轻量统计表。
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import torch
import yaml

from net.st_gcn import Model


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict[str, Any]:
    """读取 YAML 配置。

    Args:
        config_path: 配置文件路径。

    Returns:
        解析后的配置字典。
    """
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def resolve_checkpoint_path(work_dir: Path, epoch: int | None) -> Path:
    """确定要读取的 checkpoint 路径。

    Args:
        work_dir: 实验工作目录。
        epoch: 显式指定的 epoch。若为空，则自动选取目录中 epoch 编号最大的
            `*_model.pt`。

    Returns:
        checkpoint 路径。

    Raises:
        FileNotFoundError: 未找到可用 checkpoint。
    """
    if epoch is not None:
        checkpoint_path = work_dir / f"epoch{epoch}_model.pt"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"未找到指定 checkpoint: {checkpoint_path}")
        return checkpoint_path

    candidates = sorted(
        work_dir.glob("epoch*_model.pt"),
        key=lambda path: int(path.stem.replace("epoch", "").replace("_model", "")),
    )
    if not candidates:
        raise FileNotFoundError(f"目录中没有可用 checkpoint: {work_dir}")
    return candidates[-1]


def load_model_from_config(
    config: dict[str, Any],
    checkpoint_path: Path,
) -> Model:
    """按配置与 checkpoint 恢复模型。

    Args:
        config: YAML 配置内容。
        checkpoint_path: 模型权重路径。

    Returns:
        已加载权重的模型。
    """
    model = Model(**config["model_args"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict)
    model.eval()
    return model


def infer_subset_labels(strategy: str, num_subsets: int) -> list[str]:
    """推断子集标签。

    Args:
        strategy: 图策略名。
        num_subsets: 子集数量。

    Returns:
        子集标签列表。
    """
    if strategy.startswith("distance") and num_subsets == 2:
        return ["self", "neighbor"]
    if strategy.startswith("spatial") and num_subsets == 3:
        return ["root", "close", "further"]
    return [f"subset_{idx}" for idx in range(num_subsets)]


def collect_layer_stats(model: Model, strategy: str) -> list[dict[str, Any]]:
    """收集每层每个子集的有效邻接统计。

    Args:
        model: 已恢复权重的 ST-GCN 模型。
        strategy: 图策略名。

    Returns:
        统计记录列表。
    """
    base_adjacency = model.A.detach().cpu()
    subset_labels = infer_subset_labels(strategy, base_adjacency.size(0))
    rows: list[dict[str, Any]] = []

    for layer_idx, importance in enumerate(model.edge_importance, start=1):
        if isinstance(importance, torch.Tensor):
            importance_tensor = importance.detach().cpu()
        else:
            importance_tensor = torch.ones_like(base_adjacency)

        effective_adjacency = base_adjacency * importance_tensor
        for subset_idx, subset_name in enumerate(subset_labels):
            subset_mask = importance_tensor[subset_idx]
            subset_effective = effective_adjacency[subset_idx]
            rows.append(
                {
                    "layer": layer_idx,
                    "subset_index": subset_idx,
                    "subset_name": subset_name,
                    "mask_mean": float(subset_mask.mean().item()),
                    "a_eff_sum": float(subset_effective.sum().item()),
                    "a_eff_colsum_mean": float(subset_effective.sum(dim=0).mean().item()),
                    "a_eff_abs_mean": float(subset_effective.abs().mean().item()),
                }
            )

    return rows


def add_ratio_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """补充常用的 self/neighbor 比值摘要。

    当前只对 2 子集的 `distance*` 变体生成比值行，避免把 spatial 三分区
    强行压成不自然的成对比较。

    Args:
        rows: 原始逐子集统计。

    Returns:
        包含原始行与比值摘要行的新列表。
    """
    rows_by_layer: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        rows_by_layer.setdefault(int(row["layer"]), []).append(row)

    enriched_rows = list(rows)
    for layer_idx, layer_rows in rows_by_layer.items():
        subset_names = {row["subset_name"] for row in layer_rows}
        if subset_names != {"self", "neighbor"}:
            continue
        self_row = next(row for row in layer_rows if row["subset_name"] == "self")
        neighbor_row = next(row for row in layer_rows if row["subset_name"] == "neighbor")
        enriched_rows.append(
            {
                "layer": layer_idx,
                "subset_index": -1,
                "subset_name": "self_over_neighbor",
                "mask_mean": self_row["mask_mean"] / neighbor_row["mask_mean"],
                "a_eff_sum": self_row["a_eff_sum"] / neighbor_row["a_eff_sum"],
                "a_eff_colsum_mean": self_row["a_eff_colsum_mean"]
                / neighbor_row["a_eff_colsum_mean"],
                "a_eff_abs_mean": self_row["a_eff_abs_mean"]
                / neighbor_row["a_eff_abs_mean"],
            }
        )
    return enriched_rows


def write_csv(rows: list[dict[str, Any]], output_path: Path) -> None:
    """写出 CSV 结果。"""
    fieldnames = [
        "layer",
        "subset_index",
        "subset_name",
        "mask_mean",
        "a_eff_sum",
        "a_eff_colsum_mean",
        "a_eff_abs_mean",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(payload: dict[str, Any], output_path: Path) -> None:
    """写出 JSON 摘要。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    """解析命令行参数。"""
    parser = argparse.ArgumentParser(description="导出 A_eff 轻量统计")
    parser.add_argument(
        "--config",
        required=True,
        help="实验配置文件路径，例如 config/ablation/ntu-xsub/distance_imp.yaml",
    )
    parser.add_argument(
        "--work-dir",
        default=None,
        help="显式指定 checkpoint 所在工作目录；默认读取配置文件中的 work_dir",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=None,
        help="显式指定 checkpoint epoch；默认自动选择 work_dir 中最新的模型文件",
    )
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="输出前缀；默认写到 work_dir/stats/effective_adjacency",
    )
    return parser.parse_args()


def main() -> None:
    """脚本入口。"""
    args = parse_args()
    config_path = (ROOT_DIR / args.config).resolve()
    config = load_config(config_path)

    work_dir = (
        (ROOT_DIR / args.work_dir).resolve()
        if args.work_dir is not None
        else (ROOT_DIR / config["work_dir"]).resolve()
    )
    checkpoint_path = resolve_checkpoint_path(work_dir, args.epoch)
    model = load_model_from_config(config, checkpoint_path)

    strategy = str(config["model_args"]["graph_args"]["strategy"])
    rows = collect_layer_stats(model, strategy)
    rows = add_ratio_rows(rows)

    output_prefix = (
        Path(args.output_prefix).resolve()
        if args.output_prefix is not None
        else (work_dir / "stats" / "effective_adjacency")
    )
    write_csv(rows, output_prefix.with_suffix(".csv"))
    write_json(
        {
            "config": str(config_path),
            "checkpoint": str(checkpoint_path),
            "strategy": strategy,
            "rows": rows,
        },
        output_prefix.with_suffix(".json"),
    )

    print(f"config: {config_path}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"strategy: {strategy}")
    print(f"csv: {output_prefix.with_suffix('.csv')}")
    print(f"json: {output_prefix.with_suffix('.json')}")


if __name__ == "__main__":
    main()
