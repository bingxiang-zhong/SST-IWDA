import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


# %% Complex number operations

def complex_multiplication(x, y):
    return torch.stack([x[..., 0] * y[..., 0] - x[..., 1] * y[..., 1], x[..., 0] * y[..., 1] + x[..., 1] * y[..., 0]],
                       dim=-1)


def complex_conjugate_multiplication(x, y):
    return torch.stack([x[..., 0] * y[..., 0] + x[..., 1] * y[..., 1], x[..., 1] * y[..., 0] - x[..., 0] * y[..., 1]],
                       dim=-1)


def complex_cart2polar(x):
    mod = torch.sqrt(complex_conjugate_multiplication(x, x)[..., 0])
    phase = torch.atan2(x[..., 1], x[..., 0])
    return torch.stack((mod, phase), dim=-1)


# %% Signal processing and DOA estimation layers

class STFT(nn.Module):
    """ Function: Get STFT coefficients of microphone signals (batch processing by pytorch)
        Args:       win_len         - the length of frame / window
                    win_shift_ratio - the ratio between frame shift and frame length
                    nfft            - the number of fft points
                    win             - window type
                                    'boxcar': a rectangular window (equivalent to no window at all)
                                    'hann': a Hann window
                    signal          - the microphone signals in time domain (nbatch, nsample, nch)
        Returns:    stft            - STFT coefficients (nbatch, nf, nt, nch)
    """

    def __init__(self, win_len, win_shift_ratio, nfft, win='hann'):
        super(STFT, self).__init__()

        self.win_len = win_len
        self.win_shift_ratio = win_shift_ratio
        self.nfft = nfft
        self.win = win

    def forward(self, signal):
        nsample = signal.shape[-2]
        nch = signal.shape[-1]
        win_shift = int(self.win_len * self.win_shift_ratio)
        nf = int(self.nfft / 2) + 1

        nb = signal.shape[0]
        # nt = int((nsample) / win_shift) + 1  # for iSTFT
        nt = np.floor((nsample) / win_shift + 1).astype(int)
        stft = torch.zeros((nb, nf, nt, nch), dtype=torch.complex64, device=signal.device)

        if self.win == 'hann':
            window = torch.hann_window(window_length=self.win_len, device=signal.device)
        for ch_idx in range(0, nch, 1):
            stft[:, :, :, ch_idx] = torch.stft(signal[:, :, ch_idx], n_fft=self.nfft, hop_length=win_shift,
                                               win_length=self.win_len,
                                               window=window, center=True, normalized=False, return_complex=True)

        return stft



#%%

class WindowTargets:
    """Windowing transform.
    Create it indicating the window length (K), the step between windows and an optional
    window shape indicated as a vector of length K or as a Numpy window function.


    """

    def __init__(self, K, step):
        self.K = K
        self.step = step

    def __call__(self, x, acoustic_scene):
        N_mics = x.shape[1]
        N_dims = acoustic_scene["DOA"].shape[1]
        L = x.shape[0]
        N_w = np.floor(L / self.step - self.K / self.step + 1).astype(int)

        if self.K > L:
            raise Exception(
                f"The window size can not be larger than the signal length ({L})"
            )
        elif self.step > L:
            raise Exception(
                f"The window step can not be larger than the signal length ({L})"
            )

        DOAw = to_frames(acoustic_scene["DOA"], self.K, self.step)

        # Handle azimuth wrap-around (-π to π)
        for i in np.flatnonzero(np.abs(np.diff(DOAw[..., 0], axis=1)).max(axis=1) > np.pi):
            DOAw[i, DOAw[i, :, 0] < 0, 0] += 2 * np.pi  # Adjust for continuity

        # Average over windows and wrap back to [-π, π]
        DOAw = np.mean(DOAw, axis=1)  # Shape (N_w, 1)
        DOAw[DOAw[:, 0] > np.pi, 0] -= 2 * np.pi

        acoustic_scene["DOAw"] = DOAw

        # Window the VAD if it exists
        if "vad" in acoustic_scene:
            acoustic_scene["vad"] = to_frames(acoustic_scene["vad"], self.K, self.step)

        # Timestamp for each window
        acoustic_scene["tw"] = (
            np.arange(0, (L - self.K), self.step) / acoustic_scene["fs"]
        )

        # Return the original signal
        return x, acoustic_scene


def to_frames(x, frame_size, hop_size):
    """Converts a signal to frames. The first dimension of the signal is the dimension which is framed.

    Args:
        x (np.ndarray): Input signal.
        frame_size (int): Number of frames.
        hop_size (int): Step between frames.
    Returns:
        np.ndarray: Framed signal of shape (... , n_frames, frame_size)
    """

    x_shape = x.shape
    n_signal = x_shape[0]

    n_frames = int(n_signal / hop_size - frame_size / hop_size)

    n_signal = n_frames * hop_size + frame_size
    # Truncate the signal to fit an integer number of frames
    x = x[:n_signal]

    out_shape = (n_frames, frame_size) + x_shape[1:]
    x_frames = np.zeros(out_shape, dtype=x.dtype)

    for i in range(n_frames):
        x_frames[i] = x[i * hop_size : i * hop_size + frame_size]

    return x_frames


#%%

class getMetric(nn.Module):
    """
    Call:
    # single source
    getmetric = at_module.getMetric(source_mode='single', metric_unfold=True)
    metric = self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=['azi,'ele'], ae_TH=30, useVAD=False, vad_TH=vad_TH)
    """

    def __init__(self,  metric_unfold=True):
        """
        """
        super(getMetric, self).__init__()

        self.metric_unfold = metric_unfold

    def forward(self, doa_gt, vad_gt, doa_est, vad_est, ae_mode, ae_TH=30, useVAD=True, vad_TH=[0.5, 0.5]):
        """
        Args:
            doa_gt, doa_est - (nb, nt, 2, ns) in degrees
            vad_gt, vad_est - (nb, nt, ns) binary values
            ae_mode 		- angle error mode, [*, *, *], * - 'azi', 'ele', 'aziele'
            ae_TH			- angle error threshold, namely azimuth error threshold in degrees
            vad_TH 			- VAD threshold, [gtVAD_TH, estVAD_TH]
        Returns:
            ACC, MAE
        """
        device = doa_gt.device


        nbatch, nt, naziele, nsources = doa_est.shape
        if useVAD == False:
            vad_gt = torch.ones((nbatch, nt, nsources)).to(device)
            vad_est = torch.ones((nbatch, nt, nsources)).to(device)
        else:
            vad_gt = vad_gt > vad_TH[0]
            vad_est = vad_est > vad_TH[1]
        vad_est = vad_est * vad_gt

        azi_error = self.angular_error(doa_est[:, :, 1, :], doa_gt[:, :, 1, :], 'azi')
        ele_error = self.angular_error(doa_est[:, :, 0, :], doa_gt[:, :, 0, :], 'ele')
        aziele_error = self.angular_error(doa_est.permute(2, 0, 1, 3), doa_gt.permute(2, 0, 1, 3), 'aziele')

        corr_flag = ((azi_error < ae_TH) + 0.0) * vad_est  # Accorrding to azimuth error
        act_flag = 1 * vad_gt
        ACC = torch.sum(corr_flag) / torch.sum(act_flag)
        MAE = []
        if 'ele' in ae_mode:
            MAE += [torch.sum(vad_gt * ele_error) / torch.sum(act_flag)]
        elif 'azi' in ae_mode:
            MAE += [torch.sum(vad_gt * azi_error) / torch.sum(act_flag)]
        elif 'aziele' in ae_mode:
            MAE += [torch.sum(vad_gt * aziele_error) / torch.sum(act_flag)]
        else:
            raise Exception('Angle error mode unrecognized')
        MAE = torch.tensor(MAE)

        metric = [ACC, MAE]
        if self.metric_unfold:
            metric = self.unfold_metric(metric)

        return metric


    def angular_error(self, est, gt, ae_mode):
        if ae_mode == 'azi':
            ae = torch.abs((est - gt + 180) % 360 - 180)
        elif ae_mode == 'ele':
            ae = torch.abs(est - gt)
        elif ae_mode == 'aziele':
            ele_gt = gt[0, ...].float() / 180 * np.pi
            azi_gt = gt[1, ...].float() / 180 * np.pi
            ele_est = est[0, ...].float() / 180 * np.pi
            azi_est = est[1, ...].float() / 180 * np.pi
            aux = torch.cos(ele_gt) * torch.cos(ele_est) + torch.sin(ele_gt) * torch.sin(ele_est) * torch.cos(
                azi_gt - azi_est)
            aux[aux.gt(0.99999)] = 0.99999
            aux[aux.lt(-0.99999)] = -0.99999
            ae = torch.abs(torch.acos(aux)) * 180 / np.pi
        else:
            raise Exception('Angle error mode unrecognized')

        return ae

    def unfold_metric(self, metric):
        metric_unfold = []
        for m in metric:
            if m.numel() != 1:
                for n in range(m.numel()):
                    metric_unfold += [m[n]]
            else:
                metric_unfold += [m]
        return metric_unfold


class PredDOA(nn.Module):
    def __init__(self,
                 device='cuda',
                 ):
        super(PredDOA, self).__init__()
        self.getmetric = getMetric(metric_unfold=True)
        self.dev = device

    def forward(self, pred_batch, gt_batch, idx):
        pred_batch, _ = self.predgt2DOA_spect(pred_batch=pred_batch, gt_batch=gt_batch)
        metric = self.evaluate(pred_batch=pred_batch, gt_batch=gt_batch, idx=idx)
        return metric

    def predgt2DOA_spect(self, pred_batch, gt_batch):
        nb, nt, nn = pred_batch.shape
        vad_batch_pred, doa_batch_pred = pred_batch.topk(1, dim=-1)
        pred_batch = []
        pred_batch += [doa_batch_pred[:, :, :]]
        pred_batch += [vad_batch_pred]
        if gt_batch is not None:
            if type(gt_batch) is list:
                for idx in range(len(gt_batch)):
                    gt_batch[idx] = gt_batch[idx].detach()
            else:
                gt_batch = gt_batch.detach()
        return pred_batch, gt_batch


    def evaluate(self, pred_batch=None, gt_batch=None, vad_TH=[0.001, 0.05], idx=None):
        ae_mode = ['azi']
        doa_gt = torch.cat((gt_batch[0], gt_batch[0]), dim=-1)  # * 180 / np.pi

        doa_est = torch.cat((pred_batch[0], pred_batch[0]), dim=-1)
        doa_gt = doa_gt[:, :, :, np.newaxis].to(self.dev)
        doa_est = doa_est[:, :, :, np.newaxis].to(self.dev)
        vad_est = pred_batch[-1].to(self.dev)
        vad_gt = gt_batch[1].to(self.dev)

        metric_5 = {}
        # metric_10 = {}
        # print(doa_gt.shape)
        if idx != None:
            np.save('./results/' + str(idx) + '_doagt', doa_gt.cpu().numpy())
            np.save('./results/' + str(idx) + '_doaest', doa_est.cpu().numpy())
            np.save('./results/' + str(idx) + '_vadgt', vad_gt.cpu().numpy())
            np.save('./results/' + str(idx) + '_vadest', vad_est.cpu().numpy())

        # metric_10['ACC'], metric_10['MAE'], = \
        # 	self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode = ae_mode, ae_TH=10, useVAD=False, vad_TH=vad_TH)
        metric_5['ACC'], metric_5['MAE'], = \
            self.getmetric(doa_gt, vad_gt, doa_est, vad_est, ae_mode=ae_mode, ae_TH=5, useVAD=True, vad_TH=vad_TH)
        return metric_5  # , metric_10