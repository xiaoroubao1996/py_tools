import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp


def interp_func(input_mat, src_fps=25, trg_fps=62.5, expected_length=None):
    """线性插值
    将input_matrix通过线性插值的方式添加到想要的帧率上。

    :param input_mat: 输入的矩阵
    :param src_fps: 输入矩阵的fps
    :param trg_fps: 输出矩阵的fps
    :param expected_length: 如果设置则强制裁剪矩阵长度
    :return: 输出的矩阵
    """
    xp = list(np.arange(0, input_mat.shape[0], 1))
    interp_xp = list(np.arange(0, input_mat.shape[0], src_fps/trg_fps))
    if expected_length:
        while True:
            if expected_length > len(interp_xp):
                interp_xp.append(interp_xp[-1])
            elif expected_length < len(interp_xp):
                interp_xp.pop()
            else:
                break
    interp_mat = np.zeros(shape=(len(interp_xp), input_mat.shape[1]))
    for j in range(input_mat.shape[1]):
        interp_mat[:, j] = np.interp(interp_xp, xp, input_mat[:, j])
    return interp_mat


def quaternion_slerp_func(quaternion, src_fps=25, trg_fps=62.5, expected_length=None):
    """四元数球面插值
    将四元数通过球面插值的方式添加到想要的帧率上。

    :param input_mat: 输入的四元数
    :param src_fps: 输入四元数的fps
    :param trg_fps: 输出四元数的fps
    :param expected_length: 如果设置则强制裁剪矩阵长度
    :return: 输出的四元数
    """
    xp = list(np.arange(0, quaternion.shape[0], 1))
    interp_xp = list(np.arange(0, quaternion.shape[0], src_fps/trg_fps))
    while True:
        if interp_xp[-1] > xp[-1]:
            interp_xp.pop()
        else:
            break
    interp_xp.append(xp[-1])

    rots = R.from_quat(quaternion)
    slerp = Slerp(xp, rots)
    interp_rots = slerp(interp_xp).as_quat()

    if expected_length:
        while True:
            if expected_length > interp_rots.shape[0]:
                interp_rots = np.concatenate((interp_rots, interp_rots[-1:]))
            elif expected_length < interp_rots.shape[0]:
                interp_rots = interp_rots[:-1]
            else:
                break

    return interp_rots
