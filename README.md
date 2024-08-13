# Cat-Dog-Classifier
![Cat and Dog Classifier](img/thumbnail.jpg)
## Overview

This repository contains a deep learning project aimed at classifying images of cats and dogs. The project includes:

1. A custom Convolutional Neural Network (CNN) built in Pytorch.
2. A fine-tuned ResNet50 model pre-trained on Images Dataset from Kaggle.

Both models have been trained on a dataset imported from Kaggle, and the trained models along with the corresponding Jupyter notebooks are available in this repository.

## Table of Contents

- [Dataset](#dataset)
- [Models](#models)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Contributing](#contributing)

## Dataset

The dataset used for training and validation is imported from Kaggle. It contains labeled images of cats and dogs.

## Models

### Custom CNN

A custom Convolutional Neural Network (CNN) was built from scratch and trained on the dataset to classify the images into two categories: cats and dogs.

### Fine-tuned ResNet50

A pre-trained ResNet50 model, available in the PyTorch library, was fine-tuned on the dataset. The fine-tuned model is located in the "Finetuning Pretrained model" directory.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/Ich-Asadullah/Simple-ANN_cats-and-dogs_classifier_using_pytorch/
    cd Simple-ANN_cats-and-dogs_classifier_using_pytorch
    ```

## Usage

### Custom CNN

Use the `Simple-ANN_cats-and-dogs_classifier_using_pytorch.ipynb` notebook.

### Fine-tuned ResNet50

To train the fine-tuned ResNet50:

Use the `finetuning_RsNet50-CNN.ipynb` notebook.

### Loading Trained Models

To load and use the pre-trained models, refer to the following script:

```bash
checkpoint = torch.load('model_path')  # Load the saved checkpoint file
model = MyCNN().to(device)  # Create an instance of the model
# Create a new state dict that excludes unexpected keys
model_state_dict = model.state_dict()
for k in checkpoint['model_state_dict']:
    if k in model_state_dict and model_state_dict[k].shape == checkpoint['model_state_dict'][k].shape:
        model_state_dict[k] = checkpoint['model_state_dict'][k]
# Load the model state dict
model.load_state_dict(model_state_dict)
```

## Results

### Custom CNN
- Training Accuracy: 76.18%
- Validation Accuracy: 73.5%

### Fine-tuned ResNet50
- Testing Accuracy: 98.5%

## Contributing

Contributions are welcome! Please open an issue or submit a pull request if you have any improvements or new features to add.
