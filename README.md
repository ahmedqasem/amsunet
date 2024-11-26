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



<br>

## Setting Up the Environment


This repository includes a Conda environment file (`amsunet-macos.yml`) to set up the required dependencies for the project.

### Steps to Recreate the Environment


1. **Clone the Repository**  

   Clone this repository to your local machine (or download the code as a zip file):

   ```
   git clone https://github.com/ahmedqasem/amsunet.git

   cd amsunet
   ```

2. **Create the Conda Environment**  

    If you are using Mac (M-Series) run the following command 

    ```
    conda env create --file amsunet-macos.yml
    ```

    this might not work if you are on windows, if you are on Windows  ``amsunet-win.yml`` file will be added soon
    
    this code uses ``tensorflow==2.13.0``


4. **Verify the Installation**  

    ```
    conda activate amsunet-macos
    ```

    if you see ``amsunet-macos`` in the anaconda environments list then the installation was successful

    ```
    # conda environments:
    #
    base                  /path/to/conda/base
    amsunet-macos         /path/to/conda/envs/amsunet-macos

    ```


3. **Activate the Environment**  

    ```
    conda activate amsunet-macos
    ```


