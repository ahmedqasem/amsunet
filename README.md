# AMS-UNET

This repository contains the code for the AMS-UNET paper, which implements various image preprocessing, segmentation, and postprocessing techniques for DBT medical image analysis using a U-Net-based deep learning model. The repository is organized to facilitate the training, evaluation, and analysis of the model, along with the code used for calculating metric and experiments.

## Folder Structure

Here is an overview of the key files and folders in this repository:

### Folders

- **`data/`**: 
  - Contains subdirectories for storing raw and processed data
  
- **`experiments/`**: 
  - This directory is intended to store the output metrics and images generated from the experiments.

- **`sample_images/`**: 
  - Contains sample images that can be used for testing the AMS-UNET model or for demonstration purposes.

- **`trained_models/`**: 
  - Stores saved model weights or checkpoints generated during training.

### Key Python Files

- **`data.py`**: 
  - Defines data loading and augmentation functions for preprocessing images before feeding them into the model.

- **`general.py`**: 
  - Contains utility functions used across different scripts in the repository.

- **`metrics.py`**: 
  - Computes evaluation metrics for comparing model predictions with ground truth labels.

- **`model.py`**: 
  - Defines the AMS-UNET model architecture using a U-Net-based design.

- **`postprocessing.py`**: 
  - Contains functions for postprocessing model outputs, to be used in the experiments

- **`preprocessing.py`**: 
  - Provides functions for preparing images before they are input into the model.

- **`test.py`**: 
  - A testing script for evaluating the AMS-UNET model on a set of images.

- **`train.py`**: 
  - The main training script for the AMS-UNET model.

