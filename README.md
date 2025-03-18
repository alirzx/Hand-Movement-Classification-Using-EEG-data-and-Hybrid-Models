# Hand-Movement-Classification-Using-EEG-data-and-Hybrid-Models
Hand Movement Classification Using EEG data and Hybrid Models
![Hand Movement Image](https://github.com/alirzx/Hand-Movement-Classification-Using-EEG-data-and-Hybrid-Models/raw/main/HandMovement.png)


# EEG Hand Movement Classification using Deep and Hybrid Models

This repository contains the implementation of deep learning and hybrid models for classifying EEG signals related to hand movements. The project investigates the performance of standalone models such as RNN, LSTM, GRU, and CNN, as well as hybrid architectures (e.g., CNN + LSTM with attention, ResNet1D) to classify EEG data into three hand movement classes: left hand movement, right hand movement, and no movement.

## Project Overview

The objective of this project is to develop and evaluate machine learning models for classifying EEG signals captured during hand movements. The dataset comprises preprocessed EEG recordings from 14 channels, transformed into frequency domain features. Various deep learning architectures are implemented and compared to determine the most effective approach for EEG signal classification.

### Key Features
- **Dataset**: Preprocessed EEG data from hand movement experiments.
- **Models**: Standalone (CNN, RNN, LSTM, GRU) and hybrid models (e.g., CNN + LSTM with attention, ResNet1D).
- **Evaluation**: Metrics include accuracy, precision, recall, F1-score, and AUC-ROC.
- **Visualization**: Includes loss/accuracy curves, confusion matrices, and ROC curves.

## Dataset

The dataset is sourced from [EEG data from hands movement](https://www.kaggle.com/datasets/fabriciotorquato/eeg-data-from-hands-movement) on Kaggle, collected using an EMOTIV EPOC+ 14-channel EEG device with a sampling frequency of 128 Hz.

### Dataset Details
- **Channels**: 14 (AF3, F7, F3, FC5, T7, P7, O1, O2, P8, T8, FC6, F4, F8, AF4)
- **Sampling Frequency**: 128 Hz
- **Classes**: 3 (left hand movement, right hand movement, no movement)
- **Acquisition Protocol**: 
  - 25-minute sessions with cycles of visual stimuli (left arrow, right arrow, circle).
  - One scenario includes eyes-closed with audio cues.
- **Preprocessing**: 
  - Fast Fourier Transform (FFT) applied to extract 0-30 Hz frequency domain features.
  - Weighted and arithmetic means computed per channel and wave type, yielding a 14x4x2 matrix per sample (flattened to 112 features).
- **Files**: Four CSV files (`user_a.csv`, `user_b.csv`, `user_c.csv`, `user_d.csv`), each representing a different user.

For additional details, see the [original dataset repository](https://github.com/fabriciotorquato/pyxavier).

## Models

The project implements and evaluates the following models:

### Standalone Models
- **CNN**: 2D Convolutional Neural Network with max-pooling and fully connected layers.
- **RNN**: Recurrent Neural Network with two layers.
- **LSTM**: Long Short-Term Memory network with two layers.
- **GRU**: Gated Recurrent Unit network with two layers.

### Hybrid Models
- **CNN_SE**: CNN with Squeeze-and-Excitation blocks for channel-wise attention.
- **CNN1D_LSTM**: 1D CNN followed by an LSTM.
- **CNN1D_GRU**: 1D CNN followed by a GRU.
- **CNN1D_RNN**: 1D CNN followed by an RNN.
- **CNN_RNN_Attention**: 1D CNN + RNN with an attention mechanism.
- **CNN_LSTM_Attention**: 1D CNN + LSTM with an attention mechanism.
- **CNN_GRU_Attention**: 1D CNN + GRU with an attention mechanism.
- **ResNet1D**: 1D Residual Network with residual blocks.

Input data is processed as `(batch_size, 112)` for flat models or reshaped to `(batch_size, 14, 8)` for convolutional models.

## Installation

To run this project, ensure you have Python 3.7+ installed and install the required dependencies:

```bash
pip install -r requirements.txt


```

1. Data Preparation
Loading: The dataset is assumed to be loaded as a DataFrame with 11,520 rows and 114 columns (112 features, 'Class', 'user').
Normalization: Features are standardized using StandardScaler.
Label Encoding: Classes are encoded from floats (0.0, 1.0, 2.0) to integers (0, 1, 2).
Splitting: Data is split into training (70%), validation (15%), and test (15%) sets.

3. Model Training and Evaluation
Training: Models are trained with the Adam optimizer (learning rate 0.001), cross-entropy loss, and early stopping based on validation loss.
Evaluation: Metrics include loss, accuracy, precision, recall, F1-score, and AUC-ROC.
Visualization: Plots for loss/accuracy curves, confusion matrices, and ROC curves are generated.


Results
Model performance is evaluated and visualized with:

Metrics: Train/validation/test loss and accuracy, AUC-ROC, precision, recall, and F1-score (macro average).
Visualizations: Loss and accuracy curves, confusion matrices, and ROC curves.

final results from all models :

![results](https://github.com/alirzx/Hand-Movement-Classification-Using-EEG-data-and-Hybrid-Models/raw/main/results.png)
