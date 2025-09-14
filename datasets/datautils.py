"""
	Utils functions to deal with spherical coordinates in Pytorch.
"""

from math import pi
import torch
import os
import numpy as np
from numpy.linalg import norm


def circular_array_geometry(radius: float, mic_num: int) -> np.ndarray:
    pos_rcv = np.empty((mic_num, 3))
    v1 = np.array([1, 0, 0])
    v1 = normalize(v1)
    angles = np.arange(0, 2 * np.pi, 2 * np.pi / mic_num)
    for idx, angle in enumerate(angles):
        x = v1[0] * np.cos(angle) - v1[1] * np.sin(angle)
        y = v1[0] * np.sin(angle) + v1[1] * np.cos(angle)
        pos_rcv[idx, :] = normalize(np.array([x, y, 0]))
    pos_rcv *= radius
    return pos_rcv


def normalize(vec: np.ndarray) -> np.ndarray:
    # get unit vector
    vec = vec / norm(vec)
    vec = vec / norm(vec)
    assert np.isclose(norm(vec), 1), 'norm of vec is not close to 1'
    return vec


def audiowu_high_array_geometry() -> np.array:
    # the high-resolution mic array of the audio lab of westlake university
    R = 0.03
    pos_rcv = np.zeros((32, 3))
    pos_rcv[1:9, :] = circular_array_geometry(radius=R, mic_num=8)
    pos_rcv[9:17, :] = circular_array_geometry(radius=R * 2, mic_num=8)
    pos_rcv[17:25, :] = circular_array_geometry(radius=R * 3, mic_num=8)
    pos_rcv[25, :] = np.array([-R * 4, 0, 0])
    pos_rcv[26, :] = np.array([R * 4, 0, 0])
    pos_rcv[27, :] = np.array([R * 5, 0, 0])

    L = 0.045
    pos_rcv[28, :] = np.array([0, 0, L * 2])
    pos_rcv[29, :] = np.array([0, 0, L])
    pos_rcv[30, :] = np.array([0, 0, -L])
    pos_rcv[31, :] = np.array([0, 0, -L * 2])
    return pos_rcv


def search_files(dir_path, flag):
    result = []
    file_list = os.listdir(dir_path)
    for file_name in file_list:
        complete_file_name = os.path.join(dir_path, file_name)
        if os.path.isdir(complete_file_name):
            result.extend(search_files(complete_file_name, flag))
        if os.path.isfile(complete_file_name):
            if complete_file_name.endswith(flag):
                result.append(complete_file_name)
    return result


def pad_or_cut(wavs, lens, rng):
    """repeat signals if they are shorter than the length needed, then cut them to needed
    """
    for i, wav in enumerate(wavs):
        # repeat
        while len(wav) < lens[i]:
            wav = np.concatenate([wav, wav])
        # cut to needed length
        if len(wav) > lens[i]:
            start = rng.integers(low=0, high=len(wav) - lens[i] + 1)
            wav = wav[start:start + lens[i]]
        wavs[i] = wav
    return wavs


def forgetting_norm(input, sample_length=298):
    """
    Using the mean value of the near frames to normalization
    Args:
        input: feature
        sample_length: length of the training sample, used for calculating smooth factor
    Returns:
        normed feature
    Shapes:
        input: [B, C, F, T]
        sample_length_in_training: 192
    """
    assert input.ndim == 4
    batch_size, num_channels, num_freqs, num_frames = input.size()
    input = input.reshape(batch_size, num_channels * num_freqs, num_frames)

    eps = 1e-10
    mu = 0
    alpha = (sample_length - 1) / (sample_length + 1)

    mu_list = []
    for frame_idx in range(num_frames):
        if frame_idx < sample_length:
            alp = torch.min(torch.tensor([(frame_idx - 1) / (frame_idx + 1), alpha]))
            mu = alp * mu + (1 - alp) * torch.mean(
                input[:, :, frame_idx], dim=1
            ).reshape(
                batch_size, 1
            )  # [B, 1]
        else:
            current_frame_mu = torch.mean(input[:, :, frame_idx], dim=1).reshape(
                batch_size, 1
            )  # [B, 1]
            mu = alpha * mu + (1 - alpha) * current_frame_mu

        mu_list.append(mu)

        # print("input", input[:, :, idx].min(), input[:, :, idx].max(), input[:, :, idx].mean())
        # print(f"alp {idx}: ", alp)
        # print(f"mu {idx}: {mu[128, 0]}")

    mu = torch.stack(mu_list, dim=-1)  # [B, 1, T]
    # print(mu.shape)
    # output = input / (mu + eps)

    output = mu.reshape(batch_size, 1, 1, num_frames)
    return output


