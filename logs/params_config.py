import argparse


def parse_arguments():
    """Create argparse parser with all configuration arguments."""
    parser = argparse.ArgumentParser(description='CRNN Training')

    # Model parameters
    parser.add_argument('--model_name', default='crnn',
                        help='Model architecture (default: crnn)')
    parser.add_argument('--fs', type=int, default=16000,
                        help='Sampling frequency (default: 16000)')
    parser.add_argument('--speed_of_sound', type=int, default=343,
                        help='Speed of sound in m/s (default: 343)')
    parser.add_argument('--win_size', type=int, default=512,
                        help='Window size (default: 512)')
    parser.add_argument('--hop_rate', type=float, default=0.625,
                        help='Hop rate (default: 0.625)')

    # Path parameters
    parser.add_argument('--source_path_train', type=str, default='datasets/LibriSpeech/train-clean-100',
                        help='Training data path ')
    parser.add_argument('--source_path_test', type=str, default='datasets/LibriSpeech/test-clean',
                        help='Test data path ')

    parser.add_argument('--real_data_dir', type=str, default='datasets/RealMAN/',
                        help='Real recorded data path ')

    parser.add_argument('--target_path_train', type=str,
                        default='datasets/RealMAN/train/train_moving_source_location.csv',
                        help='Real recorded training data path ')

    parser.add_argument('--target_path_test', type=str,
                        default='datasets/RealMAN/test/test_moving_source_location.csv',
                        help='Real recorded testing data path ')

    parser.add_argument('--target_path_val', type=str,
                        default='datasets/RealMAN/val/val_moving_source_location.csv',
                        help='Real recorded validation data path ')

    parser.add_argument('--target_path_noise', type=str,
                        default='datasets/RealMAN/train/noise',
                        help='Real recorded noise data path ')

    parser.add_argument('--model_checkpoint_path', type=str, default='',
                        help='Model checkpoint path (default: empty)')

    # Dataset parameters
    parser.add_argument('--max_audio_len_s', type=int, default=10,
                        help='Maximum audio length in seconds (default: 10)')
    parser.add_argument('--array_test', default='realman',
                        help='Test array type (default: realman)')
    parser.add_argument('--max_rt60', type=float, default=1.0,
                        help='Maximum RT60 (default: 1.0)')
    parser.add_argument('--min_snr', type=int, default=-10,
                        help='Minimum SNR (default: -10)')
    parser.add_argument('--max_snr', type=int, default=15,
                        help='Maximum SNR (default: 15)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--nb_epochs', type=int, default=150,
                        help='Number of epochs (default: 150)')
    parser.add_argument('--nb_epoch_snr_decrease', type=int, default=30,
                        help='Number of epochs for SNR decrease (default: 30)')

    # resolution parameters
    parser.add_argument('--res_phi', type=int, default=180,
                        help='Phi resolution (default: 180)')

    args = parser.parse_args()
    return vars(args)

