"""ST-GCN 模型定义。"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from net.utils.graph import Graph
from net.utils.tgcn import ConvTemporalGraphical


class Model(nn.Module):
    """Spatial temporal graph convolutional networks。"""

    def __init__(
        self,
        in_channels: int,
        num_class: int,
        graph_args: dict,
        edge_importance_weighting: bool,
        **kwargs,
    ) -> None:
        super().__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer("A", A)

        spatial_kernel_size = A.size(0)
        temporal_kernel_size = 9
        kernel_size = (temporal_kernel_size, spatial_kernel_size)
        self.data_bn = nn.BatchNorm1d(in_channels * A.size(1))
        kwargs0 = {key: value for key, value in kwargs.items() if key != "dropout"}
        self.st_gcn_networks = nn.ModuleList(
            (
                st_gcn(in_channels, 64, kernel_size, 1, residual=False, **kwargs0),
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 64, kernel_size, 1, **kwargs),
                st_gcn(64, 128, kernel_size, 2, **kwargs),
                st_gcn(128, 128, kernel_size, 1, **kwargs),
                st_gcn(128, 128, kernel_size, 1, **kwargs),
                st_gcn(128, 256, kernel_size, 2, **kwargs),
                st_gcn(256, 256, kernel_size, 1, **kwargs),
                st_gcn(256, 256, kernel_size, 1, **kwargs),
            )
        )

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList(
                [nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks]
            )
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """执行分类前向传播。"""
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        x = F.avg_pool2d(x, x.size()[2:])
        x = x.view(N, M, -1, 1, 1).mean(dim=1)

        x = self.fcn(x)
        x = x.view(x.size(0), -1)
        return x

    def extract_feature(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """提取分类输出与中间特征。"""
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 3, 1, 2).contiguous()
        x = x.view(N * M, V * C, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, C, T)
        x = x.permute(0, 1, 3, 4, 2).contiguous()
        x = x.view(N * M, C, T, V)

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x, _ = gcn(x, self.A * importance)

        _, c, t, v = x.size()
        feature = x.view(N, M, c, t, v).permute(0, 2, 3, 4, 1)

        x = self.fcn(x)
        output = x.view(N, M, -1, t, v).permute(0, 2, 3, 4, 1)
        return output, feature


class st_gcn(nn.Module):
    """单个时空图卷积单元。"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: tuple[int, int],
        stride: int = 1,
        dropout: float = 0,
        residual: bool = True,
    ) -> None:
        super().__init__()

        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)

        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1),
                ),
                nn.BatchNorm2d(out_channels),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """执行单层 ST-GCN 前向传播。"""
        residual = self.residual(x)
        x, A = self.gcn(x, A)
        x = self.tcn(x) + residual
        return self.relu(x), A
