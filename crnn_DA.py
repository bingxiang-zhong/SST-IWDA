"""
    Trainer classes to train the models and perform inferences.
"""
from abc import ABC
from copy import deepcopy

from torchsummary import summary
from tqdm import trange

from models import Adver_network

from models.crnn import CRNN, crnnFE, Localizer
import numpy as np
import torch.nn.functional as F
import torch
import webrtcvad

from datasets.array_setup import ARRAY_SETUPS
from models.Adver_network import DomainClassifier
import models.module  as at_module
from models.grl import WarmStartGradientReverseLayer


class CRNNTrainer(ABC):
    """Trainer for models which use SRP-PHAT maps as input"""

    def __init__(self, params):
        super().__init__()
        """
        """
        self.res_phi = params["res_phi"]
        
        self.alpha_up = params["alpha_up"]
        self.max_iters = 1000
        self.gamma = params["gamma"]
        print("current parameters:", self.alpha_up, self.gamma)

        # For low resolution maps it is not possible to perform 4 cross layers
        self.model = CRNN(cnn_in_dim = 18, cnn_dim = 64, res_Phi = self.res_phi)

        # summary(model, (3, 103, 32, 64), batch_size = 1, device="cpu")
        self.discriminator = DomainClassifier()
        self.discriminator0 = DomainClassifier()
        self.FE = crnnFE()
        self.FE_t = crnnFE()
        self.Localizer = Localizer(res_Phi=self.res_phi)
        self.grl = WarmStartGradientReverseLayer(alpha=1., lo=0., hi=self.alpha_up, max_iters=self.max_iters, auto_step=True)

        checkpoint_path = params["model_checkpoint_path"]
        if checkpoint_path != "":
            self.load_checkpoint(checkpoint_path)

        self.feature_extractor = CRNNFeatureExtractor(params)

        self.get_metric = at_module.PredDOA()

        parameters = [
            {"params": self.FE_t.parameters(), "lr":  params["lr"]},  # model1 uses 0.1*lr
            {"params": self.discriminator.parameters(), "lr": params["lr"]},  # model2 uses lr
            {"params": self.Localizer.parameters(), "lr":  params["lr"]},  # model3 uses lr
            {"params": self.discriminator0.parameters(), "lr": params["lr"]}
        ]

        self.optimizer = torch.optim.Adam(parameters, lr=params["lr"])
        self.dev = 'cuda'
        self.loss_fun = torch.nn.MSELoss(reduction='none')
        self.domain_loss_func = torch.nn.BCELoss()
        self.domain_loss_func0 = torch.nn.BCELoss(reduction='none')



    def cuda(self):
        """Move the model to the GPU and perform the training and inference there."""
        self.model.cuda()
        self.discriminator.cuda()
        self.FE.cuda()
        self.FE_t.cuda()
        self.Localizer.cuda()
        self.discriminator0.cuda()
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

        fe_state_dict = {k: v for k, v in state_dict.items() if k in self.FE.state_dict()}
        localizer_state_dict = {k: v for k, v in state_dict.items() if k in self.Localizer.state_dict()}

        # Load with error and strict checking
        fe_info = self.FE.load_state_dict(fe_state_dict, strict=False)
        localizer_info = self.Localizer.load_state_dict(localizer_state_dict, strict=False)

        print(f"FE: {len(fe_state_dict)} parameters loaded. "
              f"Missing: {len(fe_info.missing_keys)}, Unexpected: {len(fe_info.unexpected_keys)}")
        print(f"Localizer: {len(localizer_state_dict)} parameters loaded. "
              f"Missing: {len(localizer_info.missing_keys)}, Unexpected: {len(localizer_info.unexpected_keys)}")

        # freeze the FE
        for param in self.FE.parameters():
            param.requires_grad = False

        self.FE_t.load_state_dict(self.FE.state_dict())



    def combine_models(self, feature_extractor, localizer, original_model):
        """
        Copies weights from split models back to original combined model
        Args:
            feature_extractor: Your trained FE module
            localizer: Your trained Localizer module
            original_model: The original combined model to update
        """
        # Get state dicts from all components
        fe_state = feature_extractor.state_dict()
        lz_state = localizer.state_dict()
        combined_state = original_model.state_dict()

        # 1. Copy FE weights (prefix handling if needed)
        for key in fe_state:
            if key in combined_state:
                combined_state[key] = fe_state[key]
            elif f'feature_extractor.{key}' in combined_state:  # Handle possible prefix
                combined_state[f'feature_extractor.{key}'] = fe_state[key]

        # 2. Copy Localizer weights
        for key in lz_state:
            if key in combined_state:
                combined_state[key] = lz_state[key]
            elif f'localizer.{key}' in combined_state:  # Handle possible prefix
                combined_state[f'localizer.{key}'] = lz_state[key]

        # 3. Load into original model
        info = original_model.load_state_dict(combined_state, strict=False)

        print(f"Combined model update: Missing keys: {info.missing_keys}, "
              f"Unexpected keys: {info.unexpected_keys}")
        return original_model


    def save_checkpoint(self, path):
        print(f"Saving model to checkpoint {path}")
        self.model = self.combine_models(self.FE_t, self.Localizer, self.model)
        torch.save(self.model.state_dict(), path)



    def extract_features(self, mic_sig_batch=None, acoustic_scene_batch=None):
        """Compute the SRP-PHAT maps from the microphone signals and extract the DoA groundtruth from the metadata dictionary."""

        output = {
            "network_input": {},
            "network_target": {},
        }

        # 1. Apply transform for mic signals
        if isinstance(mic_sig_batch, np.ndarray):
            mic_sig_batch = torch.from_numpy(mic_sig_batch.astype(np.float32))

        if self.cuda_activated:
            mic_sig_batch = mic_sig_batch.cuda()

        output["network_input"]["signal"] = self.feature_extractor(mic_sig_batch)

        # 2. Apply transform for acoustic scene
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


            # 3. Apply transform for VAD
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


    def train_epoch(self, dataset_s, dataset_t, batch_size, shuffle=True, epoch=None):
        """
        Train one epoch with domain adaptation between source and target datasets.

        Args:
            dataset_s: Source domain dataset
            dataset_t: Target domain dataset
            batch_size: Batch size for training
            shuffle: Whether to shuffle datasets
            epoch: Current epoch number for logging
        """
        # Set models to training mode
        self._set_train_mode()

        # Shuffle datasets if requested
        if shuffle:
            dataset_s.shuffle()
            dataset_t.shuffle()

        # Prepare batch iteration
        n_batches, n_batch_s, n_batch_t = self._calculate_batch_counts(
            dataset_s, dataset_t, batch_size
        )

        pbar = trange(n_batches, ascii=True)
        if epoch is not None:
            pbar.set_description(f"Epoch {epoch}")

        for i in pbar:
            # Get batches from both domains
            batch_data_s, batch_data_t = self._get_domain_batches(
                dataset_s, dataset_t, i, batch_size, n_batch_s, n_batch_t
            )

            # Forward pass and compute losses
            losses = self._compute_losses(batch_data_s, batch_data_t)

            # Backward pass
            self._update_parameters(losses)

            # Logging
            pbar.set_postfix(doa_loss=losses['doa_loss'].item())
            torch.cuda.empty_cache()


    def _set_train_mode(self):
        """Set all models to training mode."""
        self.Localizer.train()
        self.FE_t.train()
        self.discriminator.train()
        self.discriminator0.train()

    def _calculate_batch_counts(self, dataset_s, dataset_t, batch_size):
        """Calculate batch counts for both datasets."""
        n_trajectories = max(len(dataset_s), len(dataset_t))
        n_batch_s = len(dataset_s) // batch_size
        n_batch_t = len(dataset_t) // batch_size
        n_batches = n_trajectories // batch_size
        return n_batches, n_batch_s, n_batch_t

    def _get_domain_batches(self, dataset_s, dataset_t, i, batch_size, n_batch_s, n_batch_t):
        """Get and process batches from both source and target domains."""
        # Calculate batch indices with cycling
        batch_idx_s = i % n_batch_s
        batch_idx_t = i % n_batch_t

        # Get source domain batch
        batch_data_s = self._process_single_batch(
            dataset_s, batch_idx_s, batch_size, include_gt=True
        )

        # Get target domain batch
        batch_data_t = self._process_single_batch(
            dataset_t, batch_idx_t, batch_size, include_gt=False
        )

        return batch_data_s, batch_data_t

    def _process_single_batch(self, dataset, batch_idx, batch_size, include_gt=True):
        """Process a single batch from a dataset."""
        start_idx = batch_idx * batch_size
        end_idx = (batch_idx + 1) * batch_size

        # Get raw batch data
        mic_sig_batch, targets_batch = dataset.get_batch(start_idx, end_idx)

        # Extract features
        data_batch = self.extract_features(mic_sig_batch, targets_batch)
        in_batch = data_batch["network_input"]["signal"]

        # Prepare return data
        batch_data = {'input': in_batch}

        if include_gt:
            batch_data['gt'] = [
                data_batch['network_target']['doa_sph'],
                data_batch['network_target']['vad']
            ]

        return batch_data

    def _compute_losses(self, batch_data_s, batch_data_t):
        """Compute all losses for domain adaptation training."""
        self.optimizer.zero_grad()

        # Extract features from both domains
        feat_state_s, _ = self.FE(batch_data_s['input'])
        feat_state_t, _ = self.FE_t(batch_data_t['input'])

        # Compute domain losses
        domain_loss, domain_loss0 = self._compute_domain_losses(feat_state_s, feat_state_t)

        # Compute DOA loss on source domain
        doa_loss = self._compute_doa_loss(batch_data_s)

        # Total loss
        total_loss = doa_loss + domain_loss0 + self.gamma * domain_loss

        return {
            'total_loss': total_loss,
            'doa_loss': doa_loss,
            'domain_loss': domain_loss,
            'domain_loss0': domain_loss0
        }

    def _compute_domain_losses(self, feat_state_s, feat_state_t):
        """Compute domain adaptation losses."""
        batch_size_s = feat_state_s.size(0)
        batch_size_t = feat_state_t.size(0)

        # Create domain labels
        source_labels = torch.zeros(batch_size_s, 1, device=feat_state_s.device)
        target_labels = torch.ones(batch_size_t, 1, device=feat_state_t.device)
        domain_labels = torch.cat([source_labels, target_labels], dim=0)

        # Prepare features and sequence lengths
        feat_state_s_padded, feat_state_t_padded = self._align_and_concat_features(feat_state_s, feat_state_t)
        sequence_lengths = self._create_sequence_lengths(feat_state_s, feat_state_t)

        # Domain loss for discriminator (detached features)
        domain_features_detached = torch.cat([feat_state_s_padded.detach(), feat_state_t_padded.detach()], dim=0)
        domain_preds = self.discriminator(domain_features_detached, sequence_lengths)
        domain_loss = self.domain_loss_func(domain_preds, domain_labels)

        # Adversarial domain loss (with gradient reversal)
        domain_features = torch.cat([feat_state_s_padded, feat_state_t_padded], dim=0)
        domain_features_grl = self.grl(domain_features)
        domain_preds0 = self.discriminator0(domain_features_grl, sequence_lengths)

        # Compute weighted domain loss
        domain_loss0 = self._compute_weighted_domain_loss(
            domain_preds, domain_preds0, domain_labels, batch_size_s
        )

        return domain_loss, domain_loss0

    def _align_and_concat_features(self, feat_state_s, feat_state_t):
        """Align feature dimensions and concatenate."""
        max_T = max(feat_state_s.size(1), feat_state_t.size(1))

        # Pad to same length
        feat_state_s_padded = F.pad(
            feat_state_s, (0, 0, 0, max_T - feat_state_s.size(1))
        )
        feat_state_t_padded = F.pad(
            feat_state_t, (0, 0, 0, max_T - feat_state_t.size(1))
        )

        return feat_state_s_padded, feat_state_t_padded

    def _create_sequence_lengths(self, feat_state_s, feat_state_t):
        """Create sequence length tensors for both domains."""
        len_s = torch.full(
            (feat_state_s.size(0),), feat_state_s.size(1),
            dtype=torch.long
        )
        len_t = torch.full(
            (feat_state_t.size(0),), feat_state_t.size(1),
            dtype=torch.long
        )
        return torch.cat([len_s, len_t], dim=0)

    def _compute_weighted_domain_loss(self, domain_preds, domain_preds0, domain_labels, batch_size_s):
        """Compute weighted domain loss using source domain predictions."""
        # Get source domain predictions for weighting
        domain_source_preds = domain_preds[:batch_size_s]

        # Compute adaptive weights
        weight = 1 - domain_source_preds
        weight = weight / (weight.mean() + 1e-8)
        weight = weight.detach()

        # Create target weights (all ones)
        weight_t = torch.ones_like(weight, device=weight.device)
        combined_weights = torch.cat([weight, weight_t], dim=0)

        # Compute weighted loss
        domain_loss0 = self.domain_loss_func0(domain_preds0, domain_labels)
        weighted_domain_loss = (domain_loss0 * combined_weights).mean()

        return weighted_domain_loss

    def _compute_doa_loss(self, batch_data_s):
        """Compute DOA localization loss on source domain."""
        # Forward pass through target feature extractor and localizer
        feat_s, _ = self.FE_t(batch_data_s['input'])
        pred_batch_s = self.Localizer(feat_s)

        # Align dimensions
        pred_batch_s, gt_batch_s = self._align_dimensions(pred_batch_s, batch_data_s['gt'])

        # Compute classification loss
        doa_loss = self.cal_loss(pred_batch=pred_batch_s, gt_batch=gt_batch_s)
        doa_loss = torch.mean(doa_loss.view(pred_batch_s.shape[0], -1), dim=1).mean()

        return doa_loss

    def _update_parameters(self, losses):
        """Perform backward pass and parameter update."""
        losses['total_loss'].backward()
        self.optimizer.step()



    def test_epoch(self, dataset, batch_size):
        """Test the model on a dataset for one epoch."""
        self.FE_t.eval()
        self.Localizer.eval()

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
        feat, _ = self.FE_t(in_batch)
        pred_batch = self.Localizer(feat)

        # Align prediction and ground truth dimensions
        pred_batch, gt_batch = self._align_dimensions(pred_batch, gt_batch)

        # Calculate metrics
        loss = self.cal_loss(pred_batch=pred_batch, gt_batch=gt_batch).mean()
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



    def predict_step(self, batch, batch_idx: int):
        data_batch = self.extract_features(mic_sig_batch=batch)
        in_batch = data_batch['network_input']['signal']
        preds = self.model(in_batch)
        return preds[0]

    def sigmoid_entropy_loss(self, logits):
        """Compute entropy for multi-label sigmoid outputs."""
        probs = logits / logits.sum(dim=1, keepdim=True)  # Force sum=1

        # Avoid log(0) and numerical instability
        epsilon = 1e-8
        probs_clamped = torch.clamp(probs, epsilon, 1.0 - epsilon)

        entropy = -torch.sum(probs_clamped * torch.log(probs_clamped), dim=1)
        return entropy.mean()  # Average over batch

    def MSE_loss(self, preds, targets):
        nbatch = preds.shape[0]
        sum_loss = torch.nn.functional.mse_loss(preds, targets, reduction='none').contiguous().view(nbatch, -1)
        item_num = sum_loss.shape[1]
        return sum_loss.sum(axis=1) / item_num

    def cal_loss(self, pred_batch=None, gt_batch=None):
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
            angles = torch.arange(res_phi)  # .to(gt_angle)
            distance = torch.minimum(torch.abs(angles - gt_angle.item()), torch.abs(angles - gt_angle.item() + res_phi))
            out = torch.exp(-0.5 * (distance % res_phi) ** 2 / sigma ** 2)
            return out

        spectrum = torch.zeros(res_phi)  # .to(angles)
        if angles.shape[0] == 0:
            return spectrum
        for angle in angles:
            spectrum = torch.maximum(spectrum, gaussian_func_symmetric(angle, sigma)) #.cpu()
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

            # stft_rebatch = stft.to(self.dev)

            mag = torch.abs(stft)
            mean_value = torch.mean(mag.reshape(mag.shape[0], -1), dim=1)
            mean_value = mean_value[:, np.newaxis, np.newaxis, np.newaxis].expand(mag.shape)
            stft_rebatch_real = torch.real(stft) / (mean_value + eps)
            stft_rebatch_image = torch.imag(stft) / (mean_value + eps)
            real_image_batch = torch.cat(
                (stft_rebatch_real, stft_rebatch_image), dim=1)
            data = real_image_batch[:, :, self.fre_range_used, :]
            # data += [targets_batch]
        return data

