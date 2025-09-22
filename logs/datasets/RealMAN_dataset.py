# %%
import random

import matplotlib.pyplot as plt
import torch
import os

import webrtcvad
from torch.utils.data import Dataset
import soundfile as sf
import numpy as np
import pandas as pd
from scipy.io import wavfile
from scipy import signal
from datasets.datautils import search_files, audiowu_high_array_geometry
from pathlib import Path
from models.module import WindowTargets
from datasets.noise_dataset import NoiseDataset
from utils import Parameter, acoustic_power, cart2sph_np


class RealData(Dataset):
    def __init__(self, data_dir, target_dir, environment, noise_dir=None, input_fs=16000, use_mic_id=[1, 2, 3, 4, 5, 6, 7, 8, 0],
                 target_fs=16000, snr=[-10, 15],  win_size=1600, hop_rate=1, train_flag = False):
        self.ends = 'CH1.flac'
        self.data_paths = []
        self.all_targets = pd.DataFrame()
        self.env = environment
        for dir in target_dir:
            target = pd.read_csv(dir)
            self.data_paths += [data_dir + i for i in target['filename'].to_list()]
            self.all_targets = pd.concat([self.all_targets, target], ignore_index=True)

        self.all_targets.set_index('filename', inplace=True)


        if isinstance(self.env, list):
            # For list of environments - check if any environment is in the path
            self.data_paths = list(filter(
                lambda path: any(env in path for env in self.env),
                self.data_paths
            ))
            # For DataFrame filtering with multiple environments
            self.all_targets = self.all_targets[
                self.all_targets.index.str.contains('|'.join(self.env), case=False, regex=True)
            ]
        else:
            #  Single environment
            self.data_paths = list(filter(lambda path: self.env in path, self.data_paths))
            self.all_targets = self.all_targets[
                self.all_targets.index.str.contains(self.env, case=False, regex=False)
            ]

        self.train_flag = train_flag

        if self.train_flag:
            self.path_replace = 'ma_speech'
        else:
            self.path_replace = 'ma_noisy_speech'

        if noise_dir:
            self.noise_paths = search_files(noise_dir, flag=self.ends)




        self.target_fs = target_fs
        self.input_fs = input_fs
        self.SNR = snr
        self.pos_mics = audiowu_high_array_geometry()
        self.use_mic_id = use_mic_id

        # filter doa outside  0-180
        self.filtered_paths = []
        for ii in range(len(self.data_paths)):
            sig_path = self.data_paths[ii]
            azi_target = self.all_targets.at[sig_path.split('RealMAN/')[-1], 'angle(°)']
            if isinstance(azi_target, float):
                if azi_target <= 180.0:
                    self.filtered_paths.append(sig_path)
                continue
            if isinstance(azi_target, str):
                angles = [float(azi) for azi in azi_target.split(',')]
                if all(angle <= 180.0 for angle in angles):
                    self.filtered_paths.append(sig_path)

        del self.data_paths

    def __len__(self):
        return len(self.filtered_paths)

    def cal_vad(self, sig, fs=16000, th=-2.5):
        window_size = int(0.1 * fs)
        num_windows = len(sig) // window_size
        energies = []
        times = []
        for i in range(num_windows):
            window = sig[i * window_size:(i + 1) * window_size]
            fft_result = np.fft.fft(window)
            fft_result = fft_result[:window_size // 2]
            freqs = np.fft.fftfreq(window_size, 1 / fs)[:window_size // 2]
            energy = np.sum(np.abs(fft_result[(freqs >= 0) & (freqs <= 8000)]) ** 2)
            energies.append(np.log10(energy + 1e-10))
        energies = np.array(energies)
        energies = np.where(energies < th, 0, 1)

        return torch.from_numpy(energies[:,np.newaxis])


    def select_mic_array_no_circle(self, pos_mics, rng):
        mic_id_list = np.arange(28)
        specific_group_1 = {0, 2, 4, 6, 24}
        specific_group_2 = {1, 3, 5, 7, 24}
        not_use_five_linear_mics = True
        while not_use_five_linear_mics:
            num_values_to_select = rng.integers(low=2, high=9)
            CH_list = list(rng.choice(mic_id_list, num_values_to_select, replace=False))
            mic_gemo = pos_mics[CH_list, :]
            # 2 types 5-mic circle array
            if set(CH_list) == specific_group_1 or set(CH_list) == specific_group_2:
                not_use_five_linear_mics = True
            else:
                not_use_five_linear_mics = False
        return CH_list, mic_gemo

    def seg_signal(self, signal, fs, rng, dp_signal, len_signal_s=4):
        signal_start = rng.integers(low=0, high=signal.shape[0] - (len_signal_s * fs))
        # print(signal_start,signal_start*fs//frame_size,(signal_start+len_signal_s*frame_size)*fs//frame_size)
        seg_signal = signal[signal_start:signal_start + (len_signal_s * fs), :]

        seg_dp_signal = dp_signal[signal_start:signal_start + (len_signal_s * fs)]
        return seg_signal, signal_start, seg_dp_signal

    def load_signals(self, sig_path, use_mic_id):

        channels = []
        for i in use_mic_id:
            temp_path = sig_path.replace('.flac', f'_CH{i}.flac')
            single_ch_signal, fs = sf.read(temp_path)
            channels.append(single_ch_signal)
        mul_ch_signals = np.stack(channels, axis=-1)

        return mul_ch_signals, fs

    def load_noise(self, noise_path, begin_index, end_index, use_mic_id):
        channels = []

        for i in use_mic_id:
            temp_path = noise_path.replace('_CH1.flac', f'_CH{i}.flac')
            try:
                single_ch_signal, fs = sf.read(temp_path, start=begin_index, stop=end_index)
            except:
                print(temp_path, begin_index, end_index)
            channels.append(single_ch_signal)
        mul_ch_signals = np.stack(channels, axis=-1)
        return mul_ch_signals, fs

    def resample(self, mic_signal, fs, new_fs):
        signal_resampled = signal.resample(mic_signal, int(mic_signal.shape[0] * new_fs / fs))
        return signal_resampled

    def get_snr_coff(self, wav1, wav2, target_dB):
        ae1 = np.sum(wav1 ** 2) / np.prod(wav1.shape)
        ae2 = np.sum(wav2 ** 2) / np.prod(wav2.shape)
        if ae1 == 0 or ae2 == 0 or not np.isfinite(ae1) or not np.isfinite(ae2):
            return None
        coeff = np.sqrt(ae1 / ae2 * np.power(10, -target_dB / 10))
        return coeff

    def doa_interpolation(self, signal_len, doa):
        doa_len = doa.shape[0]
        original_time_indices = np.linspace(0, signal_len - 1, num=doa_len)

        # Time indices for the upsampled signal
        new_time_indices = np.arange(signal_len)  # [0, 1, 2, ..., L-1]

        # Interpolate DOA estimates
        extended_targets = np.interp(new_time_indices, original_time_indices, doa)

        return extended_targets


    def add_noise_to_signal(self, input_mic_signal, snr_item):
        """Add noise to signal by concatenating noise segments if needed"""
        signal_len = input_mic_signal.shape[0]

        required_length = signal_len
        noise_segments = []
        remaining_length = required_length

        while remaining_length > 0:
            # Randomly select noise file
            noise_id = np.random.randint(0, len(self.noise_paths))

            noise_path = self.noise_paths[noise_id]
            wav_info = sf.info(noise_path)
            source_fs = wav_info.samplerate

            # Calculate conversion ratio
            conversion_ratio = source_fs / self.target_fs

            # Convert required length to source sample rate
            required_in_source = int(np.ceil(remaining_length * conversion_ratio))
            total_frames = wav_info.frames


            # Determine how many frames we can take from this file
            available_length = min(required_in_source, total_frames)
            if available_length <= 0:
                continue  # Skip zero-length files

            # Get random segment
            max_start = total_frames - available_length
            begin_idx = np.random.randint(0, max_start)
            end_idx = begin_idx + available_length

            # Load the segment
            segment, fs = self.load_noise(noise_path,
                                          begin_index=begin_idx,
                                          end_index=end_idx,
                                          use_mic_id=self.use_mic_id)

            # Resample if needed
            if fs != self.target_fs:
                segment = self.resample(segment, fs, self.target_fs)

            usable_length = min(segment.shape[0], remaining_length)
            noise_segments.append(segment[:usable_length,:])
            remaining_length -= usable_length

        # Concatenate all segments
        if noise_segments:
            noise_signal = np.concatenate(noise_segments)

            # Trim if we got slightly more than needed (due to resampling)
            if noise_signal.shape[0] > required_length:
                noise_signal = noise_signal[:required_length, :]

            # Apply SNR
            coeff = self.get_snr_coff(input_mic_signal, noise_signal, snr_item)
            coeff = 1.0 if coeff is None else coeff
            input_mic_signal += coeff * noise_signal

        return input_mic_signal

    def __getitem__(self, idx):

        sig_path = self.filtered_paths[idx]
        load_path = sig_path.replace(self.path_replace, '')
        input_mic_signal, fs = self.load_signals(load_path, use_mic_id=self.use_mic_id)
        if fs != self.target_fs:
            input_mic_signal = self.resample(mic_signal=input_mic_signal, fs=fs, new_fs=self.target_fs)

        len_signal = input_mic_signal.shape[0]
        reference_sig = input_mic_signal[:,0]
        if self.train_flag:
          snr = self.SNR[0] + np.random.random(1) * (self.SNR[1] - self.SNR[0])
          input_mic_signal = self.add_noise_to_signal(input_mic_signal, snr)
        input_mic_signal -= input_mic_signal.mean()


        dp_sig_path = sig_path.replace(self.path_replace, 'dp_speech')
        dp_signal, dp_fs = sf.read(dp_sig_path)
        if dp_fs != self.target_fs:
            dp_signal = self.resample(mic_signal=dp_signal, fs=dp_fs, new_fs=self.target_fs)
        dp_vad = self.cal_vad(dp_signal)

        len_signal_s = len_signal / self.target_fs
        num_points = int(len_signal_s * 10)
        target = self.all_targets.at[sig_path.split('RealMAN/')[-1], 'angle(°)']
        if isinstance(target, float):
            targets = torch.ones((num_points, 1)) * int(target)
        elif isinstance(target, str):
            targets = np.array([int(float(i)) for i in target.split(',')])
            targets = torch.from_numpy(targets[:, np.newaxis])
        vad_source = torch.ones((targets.shape[0], 1))
        array_topo = self.pos_mics[self.use_mic_id]
        if vad_source.shape[0] > dp_vad.shape[0]:
            vad_source[:dp_vad.shape[0], :] = dp_vad[:, :]
        else:
            vad_source = dp_vad[:vad_source.shape[0], :]


        array_topo = {}
        array_topo['mic_pos'] = self.pos_mics[self.use_mic_id]

        acoustic_scene = {
            "array_setup": array_topo,
            "mic_pos": array_topo,
            "fs": self.target_fs,
            "DOAw": targets,
            "vad": vad_source,
            "source_signal": reference_sig,
        }


        return input_mic_signal, acoustic_scene


    def get_batch(self, idx1, idx2):
        mic_sig_batch = []
        acoustic_scene_batch = []
        for idx in range(idx1, idx2):
            mic_sig, acoustic_scene = self[idx]
            mic_sig_batch.append(mic_sig)
            acoustic_scene_batch.append(acoustic_scene)

        return self._collate_fn(mic_sig_batch, acoustic_scene_batch)

    def shuffle(self):
        random.shuffle(self.filtered_paths)

    def _collate_fn(self, mic_sig_batch, acoustic_scene_batch):
        """Collate function for the get_batch method.

        Args:
            mic_sig_batch (list): list of microphone signals (numpy arrays of shape (n_samples, n_mics)
                                                             or (n_frames, n_freq_bins, n_mics))
            acoustic_scene_batch (list): list of acoustic scenes

        Returns:
        """

        batch_size = len(mic_sig_batch)

        idx = np.argmax([sig.shape[0] for sig in mic_sig_batch])
        out_sig_shape = (batch_size,) + mic_sig_batch[idx].shape

        idx = np.argmax([scene["DOAw"].shape[0] for scene in acoustic_scene_batch])
        scene_doa_out_shape = acoustic_scene_batch[idx]["DOAw"].shape

        idx = np.argmax([scene["vad"].shape[0] for scene in acoustic_scene_batch])
        scene_vad_out_shape = acoustic_scene_batch[idx]["vad"].shape

        mic_sig_batch_out = np.zeros(out_sig_shape)
        for i in range(batch_size):
            mic_sig_batch_out[i, :mic_sig_batch[i].shape[0]] = mic_sig_batch[i]
            doaw = np.zeros(scene_doa_out_shape)
            vad = np.zeros(scene_vad_out_shape)
            nb_cur_frames = acoustic_scene_batch[i]["DOAw"].shape[0]
            doaw[:nb_cur_frames] = acoustic_scene_batch[i]["DOAw"]
            vad[:nb_cur_frames] = acoustic_scene_batch[i]["vad"]
            acoustic_scene_batch[i]["DOAw"] = doaw
            acoustic_scene_batch[i]["vad"] = vad

        return mic_sig_batch_out, np.stack(acoustic_scene_batch)


# Wrapper class for Subset
class DataWrapper(torch.utils.data.Subset):
    def __init__(self, dataset, indices):
        super().__init__(dataset, indices)
        self.dataset = dataset  # Keep reference to the original dataset

    # Forward method calls to the original dataset
    def __getattr__(self, name):
        return getattr(self.dataset, name)

