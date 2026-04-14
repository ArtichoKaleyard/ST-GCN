"""骨架图拓扑定义。"""

from __future__ import annotations

import numpy as np


class Graph:
    """描述骨架节点连接关系的图结构。

    Args:
        layout: 骨架布局名。目前保留官方实现里的 `openpose`、
            `ntu-rgb+d` 和 `ntu_edge`。
        strategy: 图分区策略。常规值为 `uniform`、`distance`、`spatial`；
            `distance_subsetnorm` 是 modern 分支里用于审计的探针策略。
        max_hop: 允许纳入邻接矩阵的最大跳数。
        dilation: 图卷积空间核的膨胀步长。

    Notes:
        原版 `distance` 策略的语义是“先对整体邻接做一次列归一化，再按 hop
        切分子集”；`distance_subsetnorm` 则相反，用于比较两种归一化口径。
    """

    def __init__(
        self,
        layout: str = "openpose",
        strategy: str = "uniform",
        max_hop: int = 1,
        dilation: int = 1,
    ) -> None:
        self.max_hop = max_hop
        self.dilation = dilation

        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop=max_hop)
        self.get_adjacency(strategy)

    def __str__(self) -> str:
        return str(self.A)

    def get_edge(self, layout: str) -> None:
        """按布局构建边关系。

        每个布局都显式包含 self-loop。`self.center` 仅供 `spatial`
        分区策略判定“离中心更近 / 更远”时使用。
        """
        if layout == "openpose":
            self.num_node = 18
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (4, 3),
                (3, 2),
                (7, 6),
                (6, 5),
                (13, 12),
                (12, 11),
                (10, 9),
                (9, 8),
                (11, 5),
                (8, 2),
                (5, 1),
                (2, 1),
                (0, 1),
                (15, 0),
                (14, 0),
                (17, 15),
                (16, 14),
            ]
            self.edge = self_link + neighbor_link
            self.center = 1
        elif layout == "ntu-rgb+d":
            self.num_node = 25
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2),
                (2, 21),
                (3, 21),
                (4, 3),
                (5, 21),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 21),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (22, 23),
                (23, 8),
                (24, 25),
                (25, 12),
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 21 - 1
        elif layout == "ntu_edge":
            self.num_node = 24
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_1base = [
                (1, 2),
                (3, 2),
                (4, 3),
                (5, 2),
                (6, 5),
                (7, 6),
                (8, 7),
                (9, 2),
                (10, 9),
                (11, 10),
                (12, 11),
                (13, 1),
                (14, 13),
                (15, 14),
                (16, 15),
                (17, 1),
                (18, 17),
                (19, 18),
                (20, 19),
                (21, 22),
                (22, 8),
                (23, 24),
                (24, 12),
            ]
            neighbor_link = [(i - 1, j - 1) for (i, j) in neighbor_1base]
            self.edge = self_link + neighbor_link
            self.center = 2
        else:
            raise ValueError("Do Not Exist This Layout.")

    def get_adjacency(self, strategy: str) -> None:
        """按分区策略构建邻接矩阵。

        `uniform` 只保留一个归一化邻接矩阵；`distance` 按 hop 切分多个
        子集但共享同一套整体归一化结果；`spatial` 则进一步按相对中心点的
        远近把每个 hop 分成 root / close / further 三类。
        """
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1

        # 官方实现对 `uniform` / `distance` / `spatial` 都先构造
        # `hop <= max_hop` 的整体邻接，再共享这一套归一化结果。
        normalize_adjacency = normalize_digraph(adjacency)

        if strategy == "uniform":
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
        elif strategy == "distance":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
        elif strategy == "distance_subsetnorm":
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                # 该分支是审计探针：先切分每个 hop 子集，再对子集单独归一化。
                subset_adjacency = np.zeros((self.num_node, self.num_node))
                subset_adjacency[self.hop_dis == hop] = 1
                A[i] = normalize_digraph(subset_adjacency)
            self.A = A
        elif strategy == "spatial":
            adjacency_list = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                if hop == 0:
                    adjacency_list.append(a_root)
                else:
                    adjacency_list.append(a_root + a_close)
                    adjacency_list.append(a_further)
            self.A = np.stack(adjacency_list)
        else:
            raise ValueError("Do Not Exist This Strategy")


def get_hop_distance(num_node: int, edge: list[tuple[int, int]], max_hop: int = 1) -> np.ndarray:
    """计算节点间跳数距离。

    返回矩阵的 `(i, j)` 元素表示节点 `i` 与 `j` 的最短 hop 距离；若在
    `max_hop` 范围内不可达，则保留为 `np.inf`。
    """
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1

    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = np.stack(transfer_mat) > 0
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    return hop_dis


def normalize_digraph(A: np.ndarray) -> np.ndarray:
    """归一化有向图邻接矩阵。

    沿列做度归一化，使每个源节点发出的总权重为 1。这与官方 ST-GCN
    对邻接矩阵的口径一致。
    """
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    return np.dot(A, Dn)


def normalize_undigraph(A: np.ndarray) -> np.ndarray:
    """归一化无向图邻接矩阵。"""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    return np.dot(np.dot(Dn, A), Dn)
