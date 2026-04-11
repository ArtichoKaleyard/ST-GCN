"""时空图卷积基础单元。"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvTemporalGraphical(nn.Module):
    """图卷积基本模块。"""

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

        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        x = torch.einsum("nkctv,kvw->nctw", (x, A))

        return x.contiguous(), A
