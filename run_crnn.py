"""
	Python script to train the CRNN model
"""

import json
import os
import sys
from datasets.RealMAN_dataset import RealData
from datasets.librispeech_dataset import LibriSpeechDataset
from datasets.random_trajectory_dataset import RandomTrajectoryDataset
from datasets.noise_dataset import NoiseDataset
import torch
from params_config import parse_arguments

from datetime import datetime

from trainers.crnn import CRNNTrainer

from utils import Parameter, set_seed
import random
import numpy as np
torch.autograd.set_detect_anomaly(True)


def _print_and_flush(msg):
    print(msg)
    sys.stdout.flush()


def main(params):
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # 1. load params
    T = params["max_audio_len_s"]
    max_rt60 = params["max_rt60"]
    min_snr = params["min_snr"]
    max_snr = params["max_snr"]

    batch_size = params["batch_size"]
    lr = params["lr"]
    nb_epoch = params["nb_epochs"]
    nb_epoch_snr_decrease = params["nb_epoch_snr_decrease"]

    model_name = params["model"]  # Only for the output filenames, change it also in Network declaration cell


    # %% Load network
    trainer = CRNNTrainer(params)
    # 4. Load dataset
    if torch.cuda.is_available():
        trainer.cuda()


    path_train = params["source_path_train"]
    path_test = params["source_path_test"]
    source_signal_dataset_train = LibriSpeechDataset(path_train, T, return_vad=True)
    source_signal_dataset_test = LibriSpeechDataset(path_test, T, return_vad=True)
    
    noise_dataset = NoiseDataset(
  	T = T,
  	fs = 16000,
  	nmic =9,
  	noise_type = 'diffuse',
  	noise_path = "datasets/NoiseX-92")

    dataset_train = RandomTrajectoryDataset(
        sourceDataset=source_signal_dataset_train,
        noiseDataset=noise_dataset,
        room_sz=Parameter([3, 3, 2.5], [10, 8, 6]),  # Random room sizes from 3x3x2.5 to 10x8x6 meters
        T60=Parameter(0.2, max_rt60) if max_rt60 > 0 else 0,
        # Random reverberation times from 0.2 to max_rt60 seconds
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),  # Random absorption weights ratios between walls
        array=params["array_test"],
        array_pos=Parameter([0.1, 0.1, 0.3], [0.9, 0.2, 0.5]),
        # Ensure a minimum separation between the array and the walls
        SNR=Parameter(15),  # Start the simulation with a low level of omnidirectional noise
        nb_points=78,  # Simulate 156 RIRs per trajectory (independent from the SRP-PHAT window length
        cache=False,
        win_size=1600,
        hop_rate=1,
        domain_label=0
    )
    dataset_test = RandomTrajectoryDataset(  # The same setup than for training but with other source signals
        sourceDataset=source_signal_dataset_test,
        noiseDataset=noise_dataset,
        room_sz=Parameter([3, 3, 2.5], [10, 8, 6]),
        T60=Parameter(0.2, max_rt60) if max_rt60 > 0 else 0,
        abs_weights=Parameter([0.5] * 6, [1.0] * 6),
        array=params["array_test"],
        array_pos=Parameter([0.1, 0.1, 0.3], [0.9, 0.2, 0.5]),
        SNR=Parameter(-10, 15),
        nb_points=78,
        win_size=1600,
        hop_rate=1,
        domain_label=0,
        cache=False
    )
    # %% Network training

    print('Training network...')

    best_test_mae = float('inf')
    start_time_str = datetime.now().strftime('%m-%d_%Hh%Mm')
    run_dir = f'logs/{model_name}_{start_time_str}'
    os.makedirs(run_dir, exist_ok=True)

    # Save params
    with open(os.path.join(run_dir, 'params.json'), 'w') as json_file:
        json.dump(params, json_file, indent=4)

    for epoch_idx in range(1, nb_epoch + 1):
        _print_and_flush('\nEpoch {}/{}:'.format(epoch_idx, nb_epoch))
        if epoch_idx == nb_epoch_snr_decrease:
            print('\nDecreasing SNR')
            # SNR between min_snr dB and 15dB after the model has started to converge
            dataset_train.SNR = Parameter(min_snr, max_snr)

        trainer.train_epoch(dataset_train, batch_size, epoch=epoch_idx)

        model_metrics = trainer.test_epoch(dataset_test, batch_size)

        _print_and_flush('Test loss: {:.4f}, '
                         'Test mae azi: {:.2f}deg,'
                         'Test acc: {:.2f}deg'.format(model_metrics['loss'],
                                                      model_metrics['mae_azi'],
                                                      model_metrics['acc']))

        if epoch_idx > 80 and model_metrics['mae_azi'] < best_test_mae:
            best_test_mae = model_metrics['mae_azi']
            _print_and_flush('New best model found at epoch {}, saving...'.format(epoch_idx))
            best_model_path = f'{run_dir}/best_ep{epoch_idx}.bin'
            trainer.save_checkpoint(best_model_path)

    print('\nTraining finished\n')

    # %% Save model
    _print_and_flush('Saving model...')

    trainer.save_checkpoint(f'{run_dir}/last.bin')
    
    trainer.load_checkpoint(best_model_path)

    # Test the model on RealMAN dataset
    target_dataset_test = RealData(data_dir='/apollo/bzh/SSL/Neural-SRP/neural_srp-main/neural_srp-main-pre/datasets/RealMAN/', target_dir=[
        '/apollo/bzh/SSL/Neural-SRP/neural_srp-main/neural_srp-main-pre/datasets/RealMAN/test/test_moving_source_location.csv'], environment='OfficeRoom3')

    model_metrics = trainer.test_epoch(target_dataset_test, batch_size)

    _print_and_flush('Test loss: {:.4f}, '
                     'Test mae azi: {:.2f}deg,'
                     'Test acc: {:.2f}deg'.format(model_metrics['loss'],
                                                  model_metrics['mae_azi'],
                                                  model_metrics['acc']))


if __name__ == "__main__":
    # Create argument parser
    params = parse_arguments()
    main(params)
