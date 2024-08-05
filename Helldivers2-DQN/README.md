# Helldivers 2 V.S. Deep Q-Networks (DQN)

This project implements an image classification system using Deep Q-Networks (DQN) with prioritized experience replay. The goal is to train a convolutional neural network to classify images into one of four categories: left, right, up, or down and then convert those classifications into in-game actions.

## Project Structure

The project consists of the following files:
- `train.py`: The main script that orchestrates the training and testing process.
- `args.py`: List of settings for the model and environment
- `model.py`: Defines the CNN architecture used for image classification.
- `network.py`: Implements the DQN algorithm with prioritized experience replay.
- `logger.py`: Handles logging of training and testing metrics using Weights and Biases (wandb).
- `utils.py`: Contains utility functions for image preprocessing and epsilon decay calculation.
- `separator.py`: Contains functions for image processing and separating relevant game elements from screenshots.

## Requirements

To run this project, you need the following dependencies:
- Python 3.12.2
- PyTorch
- NumPy
- OpenCV (cv2)
- Weights and Biases (wandb)
- PyAutoGUI
- Screen Info
- Keyboard (Optional for getting training data)
- Pillow

You can install the required packages using pip:
```
pip install torch numpy opencv-python wandb pyautogui screeninfo pillow
```

## Usage

1. Set up your Weights and Biases account and configure the project in `logger.py`.
2. Prepare your dataset by organizing the images into appropriate directories or modify the `get_random_image` function in `utils.py` to load images from your desired location.
3. Run the `main.py` script to start the training process:
   ```
   python main.py
   ```
4. Monitor the training progress using the Weights and Biases dashboard.
5. After training, the trained model checkpoint will be saved as `checkpoint.pth`.
6. To evaluate the trained model, set `load_policy = True` in `main.py` and run the script again:
   ```
   python main.py
   ```
   The testing process will load the trained model from `checkpoint.pth` and evaluate its performance on a set of test images.