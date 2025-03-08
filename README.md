# SupCon-with-EfficientNet

This repository implements Supervised Contrastive Learning (SupCon) with EfficientNet backbones for image classification.

## About

This project explores the use of SupCon loss to train EfficientNet models.

## Files

*   `dataset/`: Contains code for loading and processing datasets.
*   `losses/`:  Contains the implementation of the SupCon loss function.
*   `model/`: Contains model definitions, potentially including modifications to EfficientNet.
*   `scripts/`:  Potentially contains helper scripts for data preprocessing or model evaluation.
*   `.gitignore`: Specifies intentionally untracked files that Git should ignore.
*   `README.md`: This file.
*   `download_video.py`: Script for downloading videos (likely for creating image datasets).
*   `eval_model.py`: Script for evaluating a trained model.
*   `get_backbone.py`:  Likely contains functions for loading EfficientNet backbones.
*   `get_images.py`: Script to extract images from a video.
*   `main.py`: Main entry point for running experiments (training, evaluation, etc.).
*   `monitor.py`: Probably implements monitoring tools during training.
*   `requirements.txt`: Lists the Python packages required to run the code.
*   `save_csv.py`:  Script for saving data to CSV files.
*   `test_download.py`: Test script for video download functionality.
*   `test_error_image.py`: Test script for handling image errors.
*   `test_new_model.py`: Test script for new models.
*   `train_efficient.py`: Script for training EfficientNet models (potentially without SupCon).
*   `train_supcon.py`: Script for training EfficientNet models with SupCon loss.
*   `utils.py`:  Contains utility functions used throughout the project.

## Usage

1.  **Clone the repository:**

    ```
    git clone https://github.com/naot97/SupCon-with-EfficientNet.git
    cd SupCon-with-EfficientNet
    ```

2.  **Install the required packages:**

    ```
    pip install -r requirements.txt
    ```

3.  **Prepare your dataset:**

    *   The code expects a specific data format.  See the `dataset/` directory and the data loading functions in `train_supcon.py` and `train_efficient.py` for details.  You might need to modify the data loading code to fit your dataset.

4.  **Train the model:**

    *   To train with SupCon loss:

        ```
        python train_supcon.py --[Your options here]
        ```

    *   To train a standard EfficientNet model:

        ```
        python train_efficient.py --[Your options here]
        ```

    *   See the scripts for available training options (e.g., batch size, learning rate, etc.).

5.  **Evaluate the model:**

    ```
    python eval_model.py --[Your options here]
    ```

## Requirements

*   Python 3.x
*   PyTorch
*   `requirements.txt` lists all required packages.

## Notes

*   This `README` is a starting point.  You should add more details about the specific datasets used, the training process, and the results obtained.
*   Consider adding example commands for training and evaluation.
*   Specify how to prepare the dataset.
*   Explain the purpose of each script in more detail.
