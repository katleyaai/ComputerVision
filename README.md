# Computer Vision Assignment: Smartphone Camera Technology Comparison and MNIST Digit Recognition
This repository contains two computer vision assignments. The first assignment compares two cutting-edge smartphone camera technologies: Apple's Deep Fusion and Samsung's Adaptive Tetra-squared Pixel Sensor. The second assignment focuses on MNIST digit recognition using transfer learning with pre-trained models.

## Assignment 1: Smartphone Camera Technology Comparison

*Refer to Assignment 1 for details about camera technology comparison.*

## Assignment 2: MNIST Digit Recognition

The second assignment involves creating a Computer Vision model for a new robotic facility in East Kalimantan. The task is to teach the robot how to read a sequence of numbers, specifically identifying individual digits (0-9) in the MNIST dataset.

The robotic facility near the Titik Nol Ibu Kota Negara (IKN) Indonesia urgently requires a digit recognition model for their new robot products. Due to a tight deadline, transfer learning with pre-trained models is chosen as an efficient solution.

The MNIST dataset, a fundamental dataset for Computer Vision tasks, consists of 10 handwritten digits in grayscale (1-channel). We will experiment with pre-trained models provided by Torchvision, a sub-library of PyTorch, which were originally trained on the ImageNet dataset containing millions of RGB (3-channel) images and 1,000 classes. For simplicity, we will choose the following baseline models:
- DenseNet with modified layers to fit the 10 MNIST classes.
- Define hyperparameters and train the model with all layers trainable.
- Plot the model's performance for both training and validation results.

Additionally, we will freeze some parts of layers, namely "denseblock1" and "denseblock2," creating two separate models.
- Retrain each model, plot its performance, and examine the difference.


This assignment is implemented using Python with the following libraries:

- `torch` and `torchvision` for deep learning and computer vision.
- `time` for timing and performance measurements.
- `numpy` for random seed generation.
- `tqdm` for progress tracking.
- `matplotlib` for plotting results.
- `copy` for model cloning.
- `warnings` for filtering warnings.


To run Assignment 2 and explore MNIST digit recognition, follow these steps:

1. Open the Python script `mnist_digit_recognition.py` in the `scripts/` directory using Google Colab or your local Python environment.

2. Execute the script to experiment with the DenseNet model and plot its performance.

3. Experiment with freezing layers "denseblock1" and "denseblock2" separately by uncommenting the corresponding code sections in the script.

4. Rerun the script and analyze the performance differences.
