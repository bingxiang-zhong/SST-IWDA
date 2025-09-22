import gpuRIR
from models.module import WindowTargets
import numpy as np
import pickle
import random
import tempfile

from torch.utils.data import Dataset

from utils import Parameter, acoustic_power, cart2sph_np

from datasets.array_setup import ARRAY_SETUPS


class RandomTrajectoryDataset(Dataset):
    """Dataset Acoustic Scenes with random trajectories.
    The length of the dataset is the length of the source signals dataset.
    When you access to an element you get both the simulated signals in the microphones and a metadata dictionary.
    """

    def __init__(
            self,
            sourceDataset,
            noiseDataset,
            room_sz,
            T60,
            abs_weights,
            array,
            array_pos,
            SNR,
            nb_points,
            noise_type="omni",
            label_win_size=1600,
            label_hop_rate=1,

    ):
        """
        sourceDataset: dataset with the source signals (such as LibriSpeechDataset)
        room_sz: Size of the rooms in meters
        T60: Reverberation time of the room in seconds
        abs_weights: Absorption coefficients rations of the walls
        array: Named tuple with the characteristics of the array
        array_pos: Position of the center of the array as a fraction of the room size
        SNR: Signal to Noise Ratio
        nb_points: Number of points to simulate along the trajectory
        transforms: Transform to perform to the simulated microphone signals and the Acoustic Scene
        noise_type: "omni" or "directional" noise
        label_win_size: Window size for generating the DoA labels (default: every 1600 samples generate one label)
        label_hop_rate: Hop rate for generating the DoA labels (default: 1, no overlap in between)
        """

        self.sourceDataset = sourceDataset

        self.noiseDataset = noiseDataset

        self.shuffled_idxs = np.arange(len(sourceDataset))  # Start with unshuffled indexes

        self.array = array

        self.array_setup = ARRAY_SETUPS[self.array]
        self.N = self.array_setup["mic_pos"].shape[0]

        self.room_sz = room_sz if type(room_sz) is Parameter else Parameter(room_sz)
        self.T60 = T60 if type(T60) is Parameter else Parameter(T60)
        self.abs_weights = (
            abs_weights if type(abs_weights) is Parameter else Parameter(abs_weights)
        )
        self.array_pos = (
            array_pos if type(array_pos) is Parameter else Parameter(array_pos)
        )

        self.SNR = SNR if type(SNR) is Parameter else Parameter(SNR)
        self.nb_points = nb_points
        self.fs = sourceDataset.fs

        self.transforms = [
            WindowTargets(
                label_win_size,
                int(label_win_size * label_hop_rate),
            ),
        ]

        self.noise_type = noise_type

    def __len__(self):
        return len(self.sourceDataset)

    def __getitem__(self, idx):
        idx = self.shuffled_idxs[idx]

        acoustic_scene = self.get_random_scene(idx)
        mic_signals = simulate(acoustic_scene)

        if self.transforms is not None:
            for t in self.transforms:
                mic_signals, acoustic_scene = t(mic_signals, acoustic_scene)

            return mic_signals, acoustic_scene

    def get_batch(self, idx1, idx2):
        mic_sig_batch = []
        acoustic_scene_batch = []
        for idx in range(idx1, idx2):
            mic_sig, acoustic_scene = self[idx]  # call the get_item function
            mic_sig_batch.append(mic_sig)
            acoustic_scene_batch.append(acoustic_scene)

        return np.stack(mic_sig_batch), np.stack(acoustic_scene_batch)

    def get_random_scene(self, idx):
        # Source signal
        source_signal, vad = self.sourceDataset[idx]

        # Room
        room_sz = self.room_sz.get_value()
        T60 = self.T60.get_value()
        abs_weights = self.abs_weights.get_value()
        beta = gpuRIR.beta_SabineEstimation(room_sz, T60, abs_weights)

        # Microphones
        array_pos = self.array_pos.get_value() * room_sz
        mic_pos = array_pos + self.array_setup["mic_pos"]

        # Noises
        noise_signal = self.noiseDataset.get_random_noise(self.array_setup["mic_pos"] * (1 + 1e-5))

        # Trajectory points
        src_pos_min = np.array([0.0, 0.0, 0.0])
        src_pos_max = room_sz.copy()

        if self.array_setup["array_type"] == "planar":
            # If array is planar, make a planar trajectory in the
            # same height as the array
            src_pos_min[2] = array_pos[2]
            src_pos_max[2] = array_pos[2]

        src_pos_ini = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)
        src_pos_end = src_pos_min + np.random.random(3) * (src_pos_max - src_pos_min)

        # constraint  y > array_pos_y ---- generate the azimuth angle to be 0-180
        y_min = array_pos[1] + 0.1
        src_pos_ini[1] = y_min + np.random.random(1) * (src_pos_max[1] - y_min)
        src_pos_end[1] = y_min + np.random.random(1) * (src_pos_max[1] - y_min)

        Amax = np.min(
            np.stack(
                (
                    src_pos_ini - src_pos_min,
                    src_pos_max - src_pos_ini,
                    src_pos_end - src_pos_min,
                    src_pos_max - src_pos_end,
                )
            ),
            axis=0,
        )

        A = np.random.random(3) * np.minimum(
            Amax, 0.5  # for realman dataset using 0,5 instead of 1
        )  # Oscilations with 1m as maximum in each axis
        w = (
                2 * np.pi / self.nb_points * np.random.random(3) * 2
        )  # Between 0 and 2 oscilations in each axis ---- set to between zero and 1

        traj_pts = np.array(
            [
                np.linspace(i, j, self.nb_points)
                for i, j in zip(src_pos_ini, src_pos_end)
            ]
        ).transpose()
        traj_pts += A * np.sin(w * np.arange(self.nb_points)[:, np.newaxis])

        if np.random.random(1) < 0.25:
            traj_pts = np.ones((self.nb_points, 1)) * src_pos_ini

        # Interpolate trajectory points
        timestamps = (
                np.arange(self.nb_points) * len(source_signal) / self.fs / self.nb_points
        )
        t = np.arange(len(source_signal)) / self.fs
        trajectory = np.array(
            [np.interp(t, timestamps, traj_pts[:, i]) for i in range(3)]
        ).transpose()

        diff = trajectory - array_pos
        neg_indices = np.where(diff[:, 1] < 0)[0]
        diff[neg_indices, 1] = np.random.random(1) * 0.01

        snr = self.SNR.get_value()
        acoustic_scene = {
            "room_sz": room_sz,
            "T60": T60,
            "beta": beta,
            "SNR": snr,
            "array_setup": self.array_setup,
            "mic_pos": mic_pos,
            "source_signal": source_signal,
            "fs": self.fs,
            "t": t,
            "noise_signal": noise_signal,
            "traj_pts": traj_pts,
            "timestamps": timestamps,
            "trajectory": trajectory,
            "DOA": np.rad2deg(cart2sph_np(diff)[:, 2:3]),
            "source_vad": vad,
        }
        return acoustic_scene

    def shuffle(self):
        random.shuffle(self.shuffled_idxs)


def simulate(acoustic_scene):
    """Get the array recording using gpuRIR to perform the acoustic simulations."""
    if acoustic_scene["T60"] == 0:
        Tdiff = 0.1
        Tmax = 0.1
        nb_img = [1, 1, 1]
    else:
        Tdiff = gpuRIR.att2t_SabineEstimator(
            12, acoustic_scene["T60"]
        )  # Use ISM until the RIRs decay 12dB
        Tmax = gpuRIR.att2t_SabineEstimator(
            40, acoustic_scene["T60"]
        )  # Use diffuse model until the RIRs decay 40dB
        if acoustic_scene["T60"] < 0.15:
            Tdiff = Tmax  # Avoid issues with too short RIRs
        nb_img = gpuRIR.t2n(Tdiff, acoustic_scene["room_sz"])

    nb_mics = len(acoustic_scene["mic_pos"])
    nb_traj_pts = len(acoustic_scene["traj_pts"])
    nb_gpu_calls = min(
        int(
            np.ceil(
                acoustic_scene["fs"]
                * Tdiff
                * nb_mics
                * nb_traj_pts
                * np.prod(nb_img)
                / 1e9
            )
        ),
        nb_traj_pts,
    )
    traj_pts_batch = np.ceil(
        nb_traj_pts / nb_gpu_calls * np.arange(0, nb_gpu_calls + 1)
    ).astype(int)

    RIRs_list = [
        gpuRIR.simulateRIR(
            acoustic_scene["room_sz"],
            acoustic_scene["beta"],
            acoustic_scene["traj_pts"][traj_pts_batch[0]: traj_pts_batch[1], :],
            acoustic_scene["mic_pos"],
            nb_img,
            Tmax,
            acoustic_scene["fs"],
            Tdiff=Tdiff,
            # orV_rcv=acoustic_scene["array_setup"].mic_orV,
        )
    ]
    for i in range(1, nb_gpu_calls):
        RIRs_list += [
            gpuRIR.simulateRIR(
                acoustic_scene["room_sz"],
                acoustic_scene["beta"],
                acoustic_scene["traj_pts"][traj_pts_batch[i]: traj_pts_batch[i + 1], :],
                acoustic_scene["mic_pos"],
                nb_img,
                Tmax,
                acoustic_scene["fs"],
                Tdiff=Tdiff,
            )
        ]
    RIRs = np.concatenate(RIRs_list, axis=0)
    mic_signals = gpuRIR.simulateTrajectory(
        acoustic_scene["source_signal"],
        RIRs,
        timestamps=acoustic_scene["timestamps"],
        fs=acoustic_scene["fs"],
    )
    mic_signals = mic_signals[0: len(acoustic_scene["t"]), :]

    dp_RIRs = gpuRIR.simulateRIR(
        acoustic_scene["room_sz"],
        acoustic_scene["beta"],
        acoustic_scene["traj_pts"],
        acoustic_scene["mic_pos"],
        [1, 1, 1],
        0.1,
        acoustic_scene["fs"],
    )

    dp_signals = gpuRIR.simulateTrajectory(
        acoustic_scene["source_signal"],
        dp_RIRs,
        timestamps=acoustic_scene["timestamps"],
        fs=acoustic_scene["fs"],
    )

    ac_pow = np.mean(
        [acoustic_power(dp_signals[:, i]) for i in range(dp_signals.shape[1])]
    )

    ac_pow_noise = np.mean(
        [acoustic_power(acoustic_scene['noise_signal'][:, i]) for i in range(acoustic_scene['noise_signal'].shape[1])])
    noise_signal = np.sqrt(ac_pow / 10 ** (acoustic_scene["SNR"] / 10)) / np.sqrt(ac_pow_noise) * acoustic_scene[
        'noise_signal']
    mic_signals += noise_signal[0:len(acoustic_scene["t"]), :]

    # Apply the propagation delay to the VAD information if it exists
    if "source_vad" in acoustic_scene:
        vad = gpuRIR.simulateTrajectory(
            acoustic_scene["source_vad"],
            dp_RIRs,
            timestamps=acoustic_scene["timestamps"],
            fs=acoustic_scene["fs"],
        )
        acoustic_scene["vad"] = (
                vad[0: len(acoustic_scene["t"]), :].mean(axis=1)
                > vad[0: len(acoustic_scene["t"]), :].max() * 1e-3
        )

    return mic_signals