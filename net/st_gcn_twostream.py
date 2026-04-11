"""双流 ST-GCN 模型。"""

from __future__ import annotations

import torch
import torch.nn as nn

from .st_gcn import Model as ST_GCN


class Model(nn.Module):
    """保留原始双流结构与参数接口。"""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        self.origin_stream = ST_GCN(*args, **kwargs)
        self.motion_stream = ST_GCN(*args, **kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行双流前向传播。"""
        n, c, _t, v, m = x.size()
        zeros = x.new_zeros((n, c, 1, v, m))
        motion = torch.cat(
            (
                zeros,
                x[:, :, 1:-1] - 0.5 * x[:, :, 2:] - 0.5 * x[:, :, :-2],
                zeros,
            ),
            dim=2,
        )
        return self.origin_stream(x) + self.motion_stream(motion)
