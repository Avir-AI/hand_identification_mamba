# Biometric, Gender and Age Identification using Mamba

## Overview
This project presents a novel approach to biometric identification that integrates the efficient long-range dependency modeling of Mamba with the U-Net architecture. Our model demonstrates superior accuracy and computational efficiency compared to previous works utilizing transformers and convolutional neural networks (CNNs).

## Features
- **High Accuracy**: Outperforms traditional transformers and CNN-based models in biometric identification tasks.
- **Computational Efficiency**: Achieves better results with reduced computational overhead.
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
Clone the repository:
   ```bash
   git clone https://github.com/Avir-AI/hand_identification_mamba.git
   cd ./hand_identification_mamba
   ```
Ensure you have Python >= 3.10 installed on your system. Then, install the required libraries and dependencies.
### Requirements
```bash
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```
### Pre-trained Weights
- [Download The Model](https://drive.google.com/file/d/1Te6sOMhBtbSUTqDzwWGTPkuPiS8cdZtB/view?usp=drive_link): `best.pth`
- Move `best.pth` to: `net/pre_trained_weights`

## Inference (not implemented)
```bash
python inference.py --img_path input-path
```
## Training

To train the model, first download the necessary pre-trained weights and datasets:

1. **Pretrained Encoder Weights**: Download from [VMamba GitHub](https://github.com/MzeroMiko/VMamba/releases/download/%2320240218/vssmsmall_dp03_ckpt_epoch_238.pth)  or [google drive](https://drive.google.com/file/d/1zUczEDh09Sr2HtQclYwGBvTh0Gwydr52/view?usp=sharing) and move the file to `net/pre_trained_weights/vssmsmall_dp03_ckpt_epoch_238.pth`.
2. **Datasets**: Download the dataset of 11k hands images
   - [Download datasets](link)
   - unzip and move `datasets` directory to `./`
   
Run the training process:
```bash
python train.py
```
## Validation
After downloading the 11k hands images and , Run the validation process:
```bash
python val.py
```
## Contact
For any questions or issues, please open an issue on GitHub or contact the project maintainer at Amir.soltani1381@gmail.com.

