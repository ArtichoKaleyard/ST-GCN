"""feeder 的数据增强与统计工具。"""

from __future__ import annotations

import random

import numpy as np


def downsample(data_numpy: np.ndarray, step: int, random_sample: bool = True) -> np.ndarray:
    """时间维下采样。

    输入布局始终约定为 ``(C, T, V, M)``。
    """
    begin = np.random.randint(step) if random_sample else 0
    return data_numpy[:, begin::step, :, :]


def temporal_slice(data_numpy: np.ndarray, step: int) -> np.ndarray:
    """按固定步长切分时间维。"""
    C, T, V, M = data_numpy.shape
    return data_numpy.reshape(C, T // step, step, V, M).transpose((0, 1, 3, 2, 4)).reshape(
        C, T // step, V, step * M
    )


def mean_subtractor(data_numpy: np.ndarray, mean: float) -> np.ndarray | None:
    """对有效帧减去均值。

    只对包含有效 skeleton 的时间段做平移，尾部纯零帧保持不变。
    """
    if mean == 0:
        return None
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    end = len(valid_frame) - valid_frame[::-1].argmax()
    data_numpy[:, :end, :, :] = data_numpy[:, :end, :, :] - mean
    return data_numpy


def auto_pading(data_numpy: np.ndarray, size: int, random_pad: bool = False) -> np.ndarray:
    """不足长度时自动补零。"""
    C, T, V, M = data_numpy.shape
    if T < size:
        begin = random.randint(0, size - T) if random_pad else 0
        data_numpy_paded = np.zeros((C, size, V, M))
        data_numpy_paded[:, begin : begin + T, :, :] = data_numpy
        return data_numpy_paded
    return data_numpy


def random_choose(data_numpy: np.ndarray, size: int, auto_pad: bool = True) -> np.ndarray:
    """随机裁剪序列。"""
    _, T, _, _ = data_numpy.shape
    if T == size:
        return data_numpy
    if T < size:
        if auto_pad:
            return auto_pading(data_numpy, size, random_pad=True)
        return data_numpy
    begin = random.randint(0, T - size)
    return data_numpy[:, begin : begin + size, :, :]


def random_move(
    data_numpy: np.ndarray,
    angle_candidate: list[float] = [-10.0, -5.0, 0.0, 5.0, 10.0],
    scale_candidate: list[float] = [0.9, 1.0, 1.1],
    transform_candidate: list[float] = [-0.2, -0.1, 0.0, 0.1, 0.2],
    move_time_candidate: list[int] = [1],
) -> np.ndarray:
    """执行连续随机仿射扰动。

    与逐帧独立抖动不同，这里先在若干时间节点上采样角度、缩放和平移，
    再对整段时间做线性插值，从而得到连续变化的运动扰动。
    """
    _, T, V, M = data_numpy.shape
    move_time = random.choice(move_time_candidate)
    node = np.arange(0, T, T * 1.0 / move_time).round().astype(int)
    node = np.append(node, T)
    num_node = len(node)

    A = np.random.choice(angle_candidate, num_node)
    S = np.random.choice(scale_candidate, num_node)
    T_x = np.random.choice(transform_candidate, num_node)
    T_y = np.random.choice(transform_candidate, num_node)

    a = np.zeros(T)
    s = np.zeros(T)
    t_x = np.zeros(T)
    t_y = np.zeros(T)

    # 在相邻节点之间线性插值，复现官方“连续随机运动”而不是离散跳变。
    for i in range(num_node - 1):
        a[node[i] : node[i + 1]] = np.linspace(A[i], A[i + 1], node[i + 1] - node[i]) * np.pi / 180
        s[node[i] : node[i + 1]] = np.linspace(S[i], S[i + 1], node[i + 1] - node[i])
        t_x[node[i] : node[i + 1]] = np.linspace(T_x[i], T_x[i + 1], node[i + 1] - node[i])
        t_y[node[i] : node[i + 1]] = np.linspace(T_y[i], T_y[i + 1], node[i + 1] - node[i])

    theta = np.array(
        [[np.cos(a) * s, -np.sin(a) * s], [np.sin(a) * s, np.cos(a) * s]]
    )

    # 对每一帧的人体坐标统一施加二维仿射变换。
    for i_frame in range(T):
        xy = data_numpy[0:2, i_frame, :, :]
        new_xy = np.dot(theta[:, :, i_frame], xy.reshape(2, -1))
        new_xy[0] += t_x[i_frame]
        new_xy[1] += t_y[i_frame]
        data_numpy[0:2, i_frame, :, :] = new_xy.reshape(2, V, M)

    return data_numpy


def random_shift(data_numpy: np.ndarray) -> np.ndarray:
    """在时间维随机平移有效帧。"""
    _, T, _, _ = data_numpy.shape
    data_shift = np.zeros(data_numpy.shape)
    valid_frame = (data_numpy != 0).sum(axis=3).sum(axis=2).sum(axis=0) > 0
    begin = valid_frame.argmax()
    end = len(valid_frame) - valid_frame[::-1].argmax()

    size = end - begin
    bias = random.randint(0, T - size)
    data_shift[:, bias : bias + size, :, :] = data_numpy[:, begin:end, :, :]

    return data_shift


def openpose_match(data_numpy: np.ndarray) -> np.ndarray:
    """匹配相邻帧中的人体实例。

    算法保持官方实现语义：
    1. 先按每帧 skeleton score 给人体实例排序；
    2. 再用相邻帧关节距离做贪心匹配；
    3. 最后按整段轨迹总分重新排序。
    """
    C, T, V, M = data_numpy.shape
    assert C == 3
    score = data_numpy[2, :, :, :].sum(axis=1)
    # 每帧人体置信度排名，后续按“高分优先”做跨帧匹配。
    rank = (-score[0 : T - 1]).argsort(axis=1).reshape(T - 1, M)

    xy1 = data_numpy[0:2, 0 : T - 1, :, :].reshape(2, T - 1, V, M, 1)
    xy2 = data_numpy[0:2, 1:T, :, :].reshape(2, T - 1, V, 1, M)
    distance = ((xy2 - xy1) ** 2).sum(axis=2).sum(axis=0)

    forward_map = np.zeros((T, M), dtype=int) - 1
    forward_map[0] = range(M)
    for m in range(M):
        choose = rank == m
        forward = distance[choose].argmin(axis=1)
        for t in range(T - 1):
            distance[t, :, forward[t]] = np.inf
        forward_map[1:][choose] = forward
    assert np.all(forward_map >= 0)

    # 把逐帧匹配结果串起来，得到完整轨迹的实例索引映射。
    for t in range(T - 1):
        forward_map[t + 1] = forward_map[t + 1][forward_map[t]]

    new_data_numpy = np.zeros(data_numpy.shape)
    for t in range(T):
        new_data_numpy[:, t, :, :] = data_numpy[:, t, :, forward_map[t]].transpose(1, 2, 0)
    data_numpy = new_data_numpy

    # 最终按整段轨迹总分排序，保证输出的前几个实例尽量是主人体。
    trace_score = data_numpy[2, :, :, :].sum(axis=1).sum(axis=0)
    rank = (-trace_score).argsort()
    return data_numpy[:, :, :, rank]


def top_k_by_category(label: np.ndarray, score: np.ndarray, top_k: int) -> list[float]:
    """按类别统计 top-k 准确率。"""
    instance_num, class_num = score.shape
    rank = score.argsort()
    hit_top_k = [[] for _ in range(class_num)]
    for i in range(instance_num):
        class_index = label[i]
        hit_top_k[class_index].append(class_index in rank[i, -top_k:])

    accuracy_list = []
    for hit_per_category in hit_top_k:
        if hit_per_category:
            accuracy_list.append(sum(hit_per_category) * 1.0 / len(hit_per_category))
        else:
            accuracy_list.append(0.0)
    return accuracy_list


def calculate_recall_precision(label: np.ndarray, score: np.ndarray) -> tuple[list[float], list[float]]:
    """计算逐类召回率与精确率。"""
    instance_num, class_num = score.shape
    rank = score.argsort()
    confusion_matrix = np.zeros([class_num, class_num])

    for i in range(instance_num):
        true_l = label[i]
        pred_l = rank[i, -1]
        confusion_matrix[true_l][pred_l] += 1

    precision: list[float] = []
    recall: list[float] = []

    for i in range(class_num):
        true_p = confusion_matrix[i][i]
        false_n = sum(confusion_matrix[i, :]) - true_p
        false_p = sum(confusion_matrix[:, i]) - true_p
        precision.append(true_p * 1.0 / (true_p + false_p))
        recall.append(true_p * 1.0 / (true_p + false_n))

    return precision, recall
