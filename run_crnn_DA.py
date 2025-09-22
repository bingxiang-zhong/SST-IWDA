

"""
Training script for domain adaptation.
"""

import json
import os
import sys
from datetime import datetime
from typing import Dict, Any

import numpy as np
import torch
from utils import set_seed

from datasets.RealMAN_dataset import RealData
from datasets.librispeech_dataset import LibriSpeechDataset
from datasets.noise_dataset import NoiseDataset
from datasets.random_trajectory_dataset import RandomTrajectoryDataset
from params_config import parse_arguments
from trainers.crnn_DA import CRNNTrainer
from utils import Parameter


def print_and_flush(msg: str) -> None:
    """Print message and flush stdout."""
    print(msg)
    sys.stdout.flush()


def main():
    """Main training function."""
    set_seed(seed=42)

    # Load params
    params = parse_arguments()

    # Training configuration
    env = 'OfficeRoom3'
    max_audio_len = params["max_audio_len_s"]
    max_rt60 = params["max_rt60"]
    min_snr = params["min_snr"]
    max_snr = params["max_snr"]
    batch_size = params["batch_size"]
    learning_rate = params["lr"]
    num_epochs = 60
    snr_decrease_epoch = params["nb_epoch_snr_decrease"]
    model_name = params["model_name"]

    # Create real datasets
    target_dataset_train = RealData(
        data_dir=params['real_data_dir'],
        target_dir=[params['target_path_train']],
        environment=env,
        noise_dir=os.path.join(params['target_path_noise'], env),
        train_flag=True
    )
    target_dataset_test = RealData(
        data_dir=params['real_data_dir'],
        target_dir=[params['target_path_test']],
        environment=env
    )
    target_dataset_val = RealData(
        data_dir=params['real_data_dir'],
        target_dir=[params['target_path_val']],
        environment=env
    )
    
        # Set model parameters
    params["alpha_up"] = 0.001
    params["gamma"] = 1

    # Setup logging directory
    best_test_error = float('inf')
    start_time_str = datetime.now().strftime('%m-%d_%Hh%Mm')
    run_dir = f'logs/{model_name}_{start_time_str}_{env}_{params["alpha_up"]}_{params["gamma"]}'
    os.makedirs(run_dir, exist_ok=True)

    # Create source datasets
    source_signal_dataset_train = LibriSpeechDataset(
        params["source_path_train"], max_audio_len, return_vad=True
    )
    source_signal_dataset_test = LibriSpeechDataset(
        params["source_path_test"], max_audio_len, return_vad=True
    )

    # Create noise dataset
    noise_dataset = NoiseDataset(
        T=max_audio_len,
        fs=16000,
        nmic=9,
        noise_type='diffuse',
        noise_path="datasets/NoiseX-92"
    )

    # Create synthetic datasets
    dataset_train = RandomTrajectoryDataset(
        sourceDataset=source_signal_dataset_train,
        noiseDataset=noise_dataset,
        room_sz=Parameter([3, 3, 2.5], [10, 8, 6]),
        T60=Parameter(0.2, max_rt60) if max_rt60 > 0 else 0,
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),
        array=params["array_test"],
        array_pos=Parameter([0.1, 0.1, 0.3], [0.9, 0.2, 0.5]),
        SNR=Parameter(min_snr, max_snr),  # Start with high SNR
        nb_points=78,
    )
    dataset_test = RandomTrajectoryDataset(
        sourceDataset=source_signal_dataset_test,
        noiseDataset=noise_dataset,
        room_sz=Parameter([3, 3, 2.5], [10, 8, 6]),
        T60=Parameter(0.2, max_rt60) if max_rt60 > 0 else 0,
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),
        array=params["array_test"],
        array_pos=Parameter([0.1, 0.1, 0.3], [0.9, 0.2, 0.5]),
        SNR=Parameter(min_snr, max_snr),
        nb_points=78,
    )


    # Initialize trainer
    trainer = CRNNTrainer(params)
    trainer.load_checkpoint("logs/crnn_source.bin")
    if torch.cuda.is_available():
        trainer.cuda()

    # Training loop
    counter = 0
    best_epoch = 0
    best_model_path = None

    for epoch_idx in range(1, num_epochs + 1):

        # Train and validate
        trainer.train_epoch(dataset_train, target_dataset_train, batch_size, epoch=epoch_idx)
        model_metric = trainer.test_epoch(target_dataset_val, batch_size)

        print_and_flush(
            f'Test loss: {model_metric["loss"]:.4f}, '
            f'Test mae azi: {model_metric["mae_azi"]:.2f}deg, '
            f'Test acc: {model_metric["acc"]:.2f}deg'
        )

        # Early stopping and model saving
        if epoch_idx >= 10:
            if model_metric['mae_azi'] < best_test_error:
                best_test_error = model_metric['mae_azi']
                print_and_flush(f'New best model found at epoch {epoch_idx}, saving...')
                best_epoch = epoch_idx
                counter = 0
                best_model_path = f'{run_dir}/best_ep{best_epoch}.bin'
                trainer.save_checkpoint(best_model_path)
            else:
                counter += 1
                if counter >= 25:
                    print(f"Early stopping for model at epoch {epoch_idx}")
                    break

    # Load best model and evaluate on test set
    if best_model_path:
        trainer.load_checkpoint(best_model_path)

    model_metric = trainer.test_epoch(target_dataset_test, batch_size)

    print_and_flush(
        f'Final Test Results - '
        f'Loss: {model_metric["loss"]:.4f}, '
        f'MAE Azi: {model_metric["mae_azi"]:.2f}deg, '
        f'Acc: {model_metric["acc"]:.2f}deg'
    )

    print('\nTraining finished\n')

    # Save final metrics
    with open(f'{run_dir}/final_metrics.json', 'w') as f:
        json.dump(model_metric, f, indent=2)


if __name__ == '__main__':
    main()
