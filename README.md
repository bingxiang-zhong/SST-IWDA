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
└── utils/                 # Utility functions (seeding, parameters, etc.)
```

## 🚀 Getting Started
### 1. Requirements

Python 3.8+
PyTorch ≥ 1.10
gpuRIR
webrtcvad
Numpy, matplotlib, scipy, soundfile, pandas and tqdm

### 2. Dataset Setup

Target domain: [RealMAN dataset](https://github.com/Audio-WestlakeU/RealMAN)

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

Source domain: [LibriSpeech](https://www.openslr.org/12) (e.g., train-clean-100, test-clean)

Noise data: [NoiseX-92](https://github.com/speechdnn/Noises) or other noise dataset

Update dataset paths in params_config.py:

**
--source_path_train   # LibriSpeech training data
--source_path_test    # LibriSpeech test data
--real_data_dir       # RealMAN root directory
--target_path_*       # CSV files for train/val/test metadata
**
