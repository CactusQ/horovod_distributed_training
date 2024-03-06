# Distributed MNIST Training with TensorFlow and Horovod

This project demonstrates how to train a Deep Neural Network (DNN) on the MNIST dataset using TensorFlow and Horovod for distributed training. It's designed to scale across multiple GPUs/machines, ensuring efficient utilization of available compute resources.

## Overview

The MNIST dataset consists of 28x28 pixel grayscale images of handwritten digits (0-9), along with their corresponding labels. The goal is to train a model that can accurately classify these images into the correct digit categories.

We employ a Convolutional Neural Network (CNN) architecture, leveraging TensorFlow for model construction and training, and Horovod for distributing the training process. This approach aims to minimize training time while maximizing model accuracy.

## Features

- **Distributed Training**: Utilizes Horovod to parallelize training across multiple GPUs/machines.
- **Dynamic Batch and Epoch Configuration**: Command-line arguments allow for easy adjustments of training volumes and batch sizes.
- **MNIST Dataset Preprocessing**: Includes normalization and reshaping of images for optimal model training.
- **Customizable Model Architecture**: The CNN model can be easily adjusted or replaced to experiment with different architectures.
- **Checkpointing**: (Commented out) Support for saving model checkpoints to resume training or evaluate the model later.
- **Evaluation**: (Commented out) Code snippets for evaluating the trained model on a test dataset.

## Requirements

- Python 3.x
- TensorFlow 2.x
- Horovod
- Numpy

## Setup

Before running the training script, ensure Horovod is set up correctly for your environment. Refer to the [Horovod documentation](https://github.com/horovod/horovod) for installation instructions and setup guides.

## Usage

The training script can be run directly from the command line. To specify the total number of training batches and batch size, use the command-line arguments as follows:

```
python mnist_training_script.py [total_num_batches] [batch_size]
```

If no arguments are provided, the script will default to 100 total training batches with a batch size of 64.

## Model Architecture

The CNN model includes the following layers:

- Input Layer
- Conv2D (32 filters, 3x3 kernel)
- Conv2D (64 filters, 3x3 kernel)
- MaxPooling2D (2x2 pool size)
- Dropout (0.25)
- Flatten
- Dense (256 units, relu activation)
- Dropout (0.5)
- Dense (10 units, output layer)

## Contributing

Contributions to improve the project are welcome. Please ensure to follow the existing code structure and document any changes or additions you make.

## License

This project is open-sourced under the MIT License. See the LICENSE file for more details.
