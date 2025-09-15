## Importance-weigted Adversarial Domain Adaptation for Sound Source Tracking

This repository provides implementations of the paper "Importance-weigted Adversarial Domain Adaptation for Sound Source Tracking"

****The code includes:****

Training on simulated multi-channel audio (run_crnn.py)

Domain adaptation training using real recordings (run_crnn_DA.py)

Evaluation on real datasets (run_evaluation.py)

Configurable hyperparameters via params_config.py

## 📂 Repository Structure

```bash 
.
├── params_config.py       # Argument parser for all training/evaluation scripts
├── run_crnn.py            # Training script on simulated data
├── run_crnn_DA.py         # Domain adaptation training script
├── run_evaluation.py      # Evaluation script on real recordings
├── datasets/              # Dataset loaders (LibriSpeech, RealMAN, Noise, etc.)
├── trainers/              # CRNN trainer classes (baseline + domain adaptation)
├── models/                # CRNN model and domain discriminator Pytorch implementations
├── logs/                  # Model bin files after training
└── utils/                 # Utility functions (seeding, parameters, etc.)

```

## 🚀 Getting Started
### 1. Requirements

Python 3.8+ 

PyTorch ≥ 1.10 

[gpuRIR](https://github.com/DavidDiazGuerra/gpuRIR)

[webrtcvad](https://github.com/wiseman/py-webrtcvad) 

Numpy, matplotlib, scipy, soundfile, pandas and tqdm 


### 2. Dataset Setup

Target domain: [RealMAN dataset](https://github.com/Audio-WestlakeU/RealMAN)

The target domain data is separated into different scenes and it is organized as the follows (a bit different from the original data files in the link):

Dataset structure
```bash 
RealMAN_dataset
├── train
  ├── dp_speech
    ├── OfficeRoom3
    ├── ...
    ├── ...
  ├── OfficeRoom3
  ├── ...
  ├── ...
  ├── noise
    ├── OfficeRoom3
    ├── ...
  ├── train_moving_source_location.csv
├── test 
├── val
```

Source domain: [LibriSpeech](https://www.openslr.org/12) (train-clean-100, test-clean)

Noise data: [NoiseX-92](https://github.com/speechdnn/Noises) 


Before training, update dataset paths in params_config.py:

```bash 
--source_path_train   # LibriSpeech training data
--source_path_test    # LibriSpeech test data
--real_data_dir       # RealMAN root directory
--target_path_train/va/test       # CSV files for train/val/test metadata
```

## 🏋️ Training
****1. Train CRNN on Synthetic Data****

First train the model on source domain (synthetic data): 
```bash 
python run_crnn.py
```
To change the parameters, you can simply modify the parameters in params_config.py or call the following:
```bash 
python run_crnn_DA.py \
    --batch_size 16 \
    --lr 1e-4 \
```

****2. Domain Adaptation Training****

Then carry out the domain adversarial training using the following script, but need to specify the model checkpoint path, which is the checkpoint of the model trained with the source domain data.

```bash 
python run_crnn_DA.py --model_checkpoint_path logs/crnn_source.bin
```



