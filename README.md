# Biometric Identification using Mamba

## Overview
This project presents a novel approach to biometric identification that integrates the efficient long-range dependency modeling of Mamba with the U-Net architecture. Our model demonstrates superior accuracy and computational efficiency compared to previous works utilizing transformers and convolutional neural networks (CNNs).

## Features
- **High Accuracy**: Outperforms traditional transformers and CNN-based models in biometric identification tasks.
- **Computational Efficiency**: Achieves better results with reduced computational overhead, making it suitable for real-time applications.
- **Innovative Architecture**: Combines the strengths of Mamba for long-range dependency modeling with the U-Net architecture for detailed spatial feature extraction.
- **Pretrained Weights**: Includes pretrained weights for faster deployment and fine-tuning.

## Comparison to Previous Works
### Transformers
- **Accuracy**: Our model achieves higher accuracy rates compared to transformer-based models, particularly in complex biometric datasets.
- **Efficiency**: Our model is computationally more efficient, reducing the required computational resources and inference time, making it more practical for real-world applications.

### CNNs
- **Performance**: Traditional CNN-based models, while effective, fall short in accuracy when compared to our approach.
- **Results**: The integration of Mamba's long-range dependency modeling with U-Net’s spatial feature extraction allows the model to capture finer details, leading to better identification results.

## Getting Started

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Avir-AI/hand_identification_mamba.git
   cd biometric-identification-SU-SUM
   Ensure you have Python >= 3.10 installed on your system. Then, install the required libraries and dependencies.
   ```
### Requirements
```bash
pip install -r requirements.txt
```
### Pre-trained Weights
- [Download The Model](https://drive.google.com/file/d/1HYkJykldDl1DiKmvkk6T3j8IM3qI1U6G/view?usp=drive_link): `new_best.pth`
- Move `new_best.pth` to: `net/pre_trained_weights`

## Training

To train the model, first download the necessary pre-trained weights and datasets:

1. **Pretrained Encoder Weights**: Download from [VMamba GitHub](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmsmall_dp03_ckpt_epoch_238.pth)  or [google drive](https://drive.google.com/file/d/1zUczEDh09Sr2HtQclYwGBvTh0Gwydr52/view?usp=sharing) and move the file to `net/pre_trained_weights/vssmsmall_dp03_ckpt_epoch_238.pth`.
2. **Datasets**: Download the dataset of 7 different sets from the provided Google Drive link. This zip file contains 256x256 images of stimuli, saliency maps, fixation maps, and ID CSVs of datasets SALICON, MIT1003, CAT2000, SALECI, UEYE, and FIWI.
   - [Download datasets](https://drive.google.com/file/d/1Mdk97UB0phYDZv8zgjBayeC1I1_QcUmh/view?usp=drive_link)
   - unzip and move `datasets` directory to `./`
   
Run the training process:

```bash
python train.py
```
## Validation

For model validation on the dataset's validation set, download the dataset as mentioned above. then execute the validation script:

```bash
python validation.py
```

