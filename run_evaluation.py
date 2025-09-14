#%%
import json
import sys

import torch

from datasets.RealMAN_dataset import RealData

from trainers.crnn import CRNNTrainer

from params_config import parse_arguments

def _print_and_flush(msg):
    print(msg)
    sys.stdout.flush()

if __name__ == '__main__':
    params =parse_arguments()
    # Parse arguments
    params['model_checkpoint_path'] = 'logs/crnn_08-07_14h15m/best_ep134.bin'

    trainer = CRNNTrainer(params)
    if torch.cuda.is_available():
        trainer.cuda()

    target_dataset_test = RealData(data_dir='/apollo/bzh/SSL/Neural-SRP/neural_srp-main/neural_srp-main-pre/datasets/RealMAN/', target_dir=[
        '/apollo/bzh/SSL/Neural-SRP/neural_srp-main/neural_srp-main-pre/datasets/RealMAN/test/test_moving_source_location.csv'], environment='OfficeRoom3')

    model_metrics = trainer.test_epoch(target_dataset_test, params['batch_size'])

    _print_and_flush('Test loss: {:.4f}, '
                     'Test mae azi: {:.2f}deg,'
                     'Test acc: {:.2f}deg'.format(model_metrics['loss'],
                                                  model_metrics['mae_azi'],
                                                  model_metrics['acc']))
