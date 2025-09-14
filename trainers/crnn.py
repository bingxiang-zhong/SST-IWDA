"""
    Trainer classes to train the models and perform inferences.
"""
from abc import ABC

from torchsummary import summary
from tqdm import trange

from models.crnn import CRNN
import numpy as np
import torch
import webrtcvad
import models.module  as at_module


class CRNNTrainer(ABC):

    def __init__(self, params):
        super().__init__()

        self.res_phi = params["res_phi"]

        # For low resolution maps it is not possible to perform 4 cross layers
        self.model = CRNN(cnn_in_dim = 2*9, cnn_dim = 64, res_Phi = self.res_phi)

        checkpoint_path = params["model_checkpoint_path"]

        if checkpoint_path != "":
            self.load_checkpoint(checkpoint_path)

        self.feature_extractor = CRNNFeatureExtractor(params)

        self.get_metric = at_module.PredDOA()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=params["lr"])
        self.dev = 'cuda'
        self.loss_fun = torch.nn.MSELoss()


    def cuda(self):
        """Move the model to the GPU and perform the training and inference there."""
        self.model.cuda()
        self.cuda_activated = True

        if self.feature_extractor is not None:
            print('feature extractor activated')
            self.feature_extractor.cuda()

    def cpu(self):
        """Move the model back to the CPU and perform the training and inference here."""
        self.model.cpu()
        self.cuda_activated = False

    def forward(self, batch):
        if self.feature_extractor is not None:
            batch = self.feature_extractor(batch)
        return self.model(batch)

    def load_checkpoint(self, path):
        print(f"Loading model from checkpoint {path}")
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.load_state_dict(state_dict)

    def save_checkpoint(self, path):
        print(f"Saving model to checkpoint {path}")
        torch.save(self.model.state_dict(), path)


    def extract_features(self, mic_sig_batch=None, acoustic_scene_batch=None):
        output = { "network_input": {}, "network_target": {}}

        if isinstance(mic_sig_batch, np.ndarray):
            mic_sig_batch = torch.from_numpy(mic_sig_batch.astype(np.float32))

        if self.cuda_activated:
            mic_sig_batch = mic_sig_batch.cuda()

        output["network_input"]["signal"] = self.feature_extractor(mic_sig_batch)


        if acoustic_scene_batch is not None:
            DOA_batch = torch.from_numpy(
                np.stack(
                    [
                        acoustic_scene_batch[i]["DOAw"].astype(np.float32)
                        for i in range(len(acoustic_scene_batch))
                    ]
                )
            )
            if self.cuda_activated:
                DOA_batch = DOA_batch.cuda()
            output["network_target"]["doa_sph"] = DOA_batch

            if "vad" in acoustic_scene_batch[0]:
                vad_batch = np.stack(
                    [
                        acoustic_scene_batch[i]["vad"]
                        for i in range(len(acoustic_scene_batch))
                    ]
                )

                # Remove last dimension
                vad_batch = vad_batch.mean(axis=2) > 2 / 3

                vad_batch = torch.from_numpy(vad_batch)  # boolean
                if self.cuda_activated:
                    vad_batch = vad_batch.cuda()

                output["network_target"]["vad"] = vad_batch.unsqueeze(-1)
            else:
                # Create a dummy, always on VAD
                doa_sph = output["network_target"]["doa_sph"]
                output["network_target"]["vad"] = torch.ones(doa_sph.shape[:2]).to(
                    doa_sph.device
                )

        return output

    def train_epoch(self, dataset,  batch_size, shuffle=True, epoch=None):
        self.model.train()
        if shuffle:
            dataset.shuffle()
        n_trajectories = len(dataset)
        pbar = trange(n_trajectories // batch_size, ascii=True)

        for i in pbar:
            if epoch is not None:
                pbar.set_description("Epoch {}".format(epoch))

            mic_sig_batch, targets_batch = dataset.get_batch(i  * batch_size,(i + 1) * batch_size,)
            data_batch = self.extract_features(mic_sig_batch, targets_batch)
            in_batch = data_batch["network_input"]["signal"]
            gt_batch = [data_batch['network_target']['doa_sph'], data_batch['network_target']['vad']]

            self.optimizer.zero_grad()

            pred_batch = self.model(in_batch)
            pred_batch, gt_batch = self._align_dimensions(pred_batch, gt_batch)
            loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)

            loss.backward()
            self.optimizer.step()

            log_msg = {"doa_loss": loss.item()}
            pbar.set_postfix(**log_msg)
            torch.cuda.empty_cache()

    def test_epoch(self, dataset, batch_size):
        """Test the model on a dataset for one epoch."""
        self.model.eval()

        total_samples = 0
        model_metric = {'loss': 0, 'mae_azi': 0, 'acc': 0}
        n_trajectories = len(dataset)

        with torch.no_grad():
            # Process all batches
            for start_idx in trange(0, n_trajectories, batch_size):
                end_idx = min(start_idx + batch_size, n_trajectories)
                batch_metrics = self._process_batch(dataset, start_idx, end_idx)

                # Accumulate weighted metrics
                nb_samples = batch_metrics['nb_samples']
                model_metric['loss'] += batch_metrics['loss'] * nb_samples
                model_metric['mae_azi'] += batch_metrics['mae_azi'] * nb_samples
                model_metric['acc'] += batch_metrics['acc'] * nb_samples
                total_samples += nb_samples

        # Average the metrics
        for key in model_metric:
            model_metric[key] /= total_samples

        return model_metric

    def _process_batch(self, dataset, start_idx, end_idx):
        """Process a single batch and return metrics."""
        # Get batch data
        mic_sig_batch, targets_batch = dataset.get_batch(start_idx, end_idx)
        data_batch = self.extract_features(mic_sig_batch, targets_batch)

        # Prepare inputs and targets
        in_batch = data_batch["network_input"]["signal"]
        gt_batch = [
            data_batch['network_target']['doa_sph'],
            data_batch['network_target']['vad']
        ]
        nb_samples = in_batch.shape[0]

        # Forward pass
        pred_batch = self.model(in_batch)

        # Align prediction and ground truth dimensions
        pred_batch, gt_batch = self._align_dimensions(pred_batch, gt_batch)

        # Calculate metrics
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch)
        metric = self.get_metric(
            pred_batch=pred_batch,
            gt_batch=gt_batch,
            idx=None
        )

        return {
            'loss': loss.item(),
            'mae_azi': metric['MAE'].item(),
            'acc': metric['ACC'].item(),
            'nb_samples': nb_samples
        }


    def _align_dimensions(self, pred_batch, gt_batch):
        """
        Align dimensions of prediction and ground truth batch.
        """
        if pred_batch.shape[1] > gt_batch[0].shape[1]:
            pred_batch = pred_batch[:, :gt_batch[0].shape[1], :]
        else:
            gt_batch[0] = gt_batch[0][:, :pred_batch.shape[1], :]
            gt_batch[1] = gt_batch[1][:, :pred_batch.shape[1], :]
        return pred_batch, gt_batch


    def cal_loss(self, pred_batch=None, gt_batch=None):
        """
        Calculate loss of prediction and Gaussian encoded ground truth batch.
        """
        doa_batch = gt_batch[0]
        vad_batch = gt_batch[1]
        doa_batch = doa_batch[:, :, :].type(torch.LongTensor).cuda()
        nb, nt, _ = pred_batch.shape
        new_target_batch = torch.zeros(nb, nt, self.res_phi)

        for b in range(nb):
            for t in range(nt):
                new_target_batch[b, t, :] = self.gaussian_encode_symmetric(angles=doa_batch[b, t,],
                                                                           res_phi=self.res_phi)
        vad_expanded = vad_batch.expand(-1, -1, self.res_phi)
        new_target_batch = new_target_batch * vad_expanded.to(new_target_batch)

        pred_batch_cart = pred_batch.to(self.dev)
        new_target_batch = new_target_batch.to(self.dev)

        loss = self.loss_fun(
            pred_batch_cart, new_target_batch)

        return loss

    def gaussian_encode_symmetric(self, angles, res_phi, sigma=16):
        def gaussian_func_symmetric(gt_angle, sigma):
            angles = torch.arange(res_phi)
            distance = torch.minimum(torch.abs(angles - gt_angle.item()), torch.abs(angles - gt_angle.item() + res_phi))
            out = torch.exp(-0.5 * (distance % res_phi) ** 2 / sigma ** 2)
            return out

        spectrum = torch.zeros(res_phi)
        if angles.shape[0] == 0:
            return spectrum
        for angle in angles:
            spectrum = torch.maximum(spectrum, gaussian_func_symmetric(angle, sigma))
        return spectrum



class CRNNFeatureExtractor(torch.nn.Module):
    def __init__(self, params):
        super().__init__()
        win_size = params["win_size"]
        hop_rate = params["hop_rate"]

        self.c = params["speed_of_sound"]
        self.fs = params["fs"]

        self.nfft = win_size
        self.res_phi = params["res_phi"]
        self.dostft = at_module.STFT(
            win_len=win_size, win_shift_ratio=hop_rate, nfft=win_size)
        self.fre_range_used = range(1, int(self.nfft / 2) + 1, 1)


    def forward(self, mic_sig_batch=None, targets_batch=None, eps=1e-6):
        if mic_sig_batch is not None:
            mic_sig_batch = mic_sig_batch
            stft = self.dostft(signal=mic_sig_batch)
            nb, nf, nt, nc = stft.shape
            stft = stft.permute(0, 3, 1, 2)
            mag = torch.abs(stft)
            mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
            mean_value = mean_value[:, np.newaxis, np.newaxis, np.newaxis].expand(mag.shape)
            stft_real = torch.real(stft) / (mean_value + eps)
            stft_image = torch.imag(stft) / (mean_value + eps)
            real_image_batch = torch.cat(
                (stft_real, stft_image), dim=1)
            data = real_image_batch[:, :, self.fre_range_used, :]

        return data

