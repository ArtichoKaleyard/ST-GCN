"""时空图卷积基础单元。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    """图卷积基本模块。

    Args:
        in_channels: 输入通道数。
        out_channels: 输出通道数。
        kernel_size: 图卷积空间核大小，即邻接子集数 `K`。
        t_kernel_size: 时间卷积核大小。
        t_stride: 时间卷积步幅。
        t_padding: 时间维 padding。
        t_dilation: 时间卷积膨胀系数。
        bias: 是否启用卷积偏置。

    Shape:
        - Input[0]: ``(N, in_channels, T_in, V)``
        - Input[1]: ``(K, V, V)``
        - Output[0]: ``(N, out_channels, T_out, V)``
        - Output[1]: ``(K, V, V)``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        t_kernel_size: int = 1,
        t_stride: int = 1,
        t_padding: int = 0,
        t_dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(
            in_channels,
            out_channels * kernel_size,
            kernel_size=(t_kernel_size, 1),
            padding=(t_padding, 0),
            stride=(t_stride, 1),
            dilation=(t_dilation, 1),
            bias=bias,
        )

    def forward(self, x: torch.Tensor, A: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """执行图卷积，保持旧版张量语义。"""
        assert A.size(0) == self.kernel_size

        x = self.conv(x)

        # 先把卷积输出拆成 `K` 个子集，再用爱因斯坦求和按邻接矩阵聚合。
        # 这里的 reshape / einsum 顺序直接决定了与官方实现的数值等价性。
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))

        return x.contiguous(), A
