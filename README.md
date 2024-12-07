# Neural Network from Scratch

This repository contains a Python implementation of a feedforward neural network built entirely from scratch using NumPy. The network supports multiple layers, customizable neurons per layer, and common activation functions, enabling experimentation and learning of neural network fundamentals.

## Features

- Load and preprocess datasets (supports MNIST-like image datasets).
- Customizable network architecture with variable layers and neurons.
- Forward and backward propagation implemented from scratch.
- Support for common activation functions: sigmoid, tanh, ReLU, and softmax.
- Training with mini-batches and visualization of loss and accuracy.
- Visualize hidden layer representations with t-SNE.
- Save and load trained models using pickle.
- Confusion matrix plotting for classification analysis.

## Getting Started

### Prerequisites

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn
- Pickle

You can install the required packages with:
```bash
pip install numpy matplotlib scikit-learn
```

### Dataset

The implementation is designed for a dataset structured similarly to MNIST, with image files organized in folders. The folder structure should look like this:
```
Task3_MNIST_Data/
    train/
        0/
        1/
        ...
        9/
    test/
        0/
        1/
        ...
        9/
```

Each subdirectory (0-9) contains grayscale images of corresponding class labels.

### Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/neural-network-from-scratch.git
   cd neural-network-from-scratch
   ```

2. Ensure the dataset is structured and available in the `Task3_MNIST_Data` folder.

3. Run the main method to train and evaluate the model:
   ```bash
   python main.py
   ```

### Key Functions

#### Dataset Loading and Preprocessing
- `loadDataset(path)` - Loads images and labels from the dataset folder.
- `meanSubtraction(images)` - Performs mean subtraction for preprocessing.

#### Neural Network Operations
- `initialize_parameters(input_dim, neurons_per_layer, no_of_layers)` - Initializes weights and biases.
- `forward_propagation(x, parameters, forward_cache)` - Implements forward propagation.
- `backward_prop(x, y, parameters, forward_cache, gradients)` - Implements backward propagation.
- `update_parameters(parameters, gradients, learning_rate)` - Updates weights and biases using gradient descent.

#### Training and Evaluation
- `train(epochs, x, y, parameters, forward_cache, gradients, learning_rate)` - Trains the neural network.
- `predict(x, parameters, forward_cache)` - Makes predictions for input data.
- `accuracy(pred_values, target_values)` - Calculates accuracy.

#### Visualization
- `plot_loss_acc(epochs, cost_list, accuracy_list)` - Plots training loss and accuracy.
- `visualize_data_tsne(dataset_x, dataset_y)` - Visualizes data using t-SNE.
- `plot_confusion_matrix(Y, predicted_values)` - Plots a confusion matrix.

#### Model Persistence
- `saveModel(name, model)` - Saves the trained model to a file.
- `loadModel(name)` - Loads a model from a file.

## Customization

- Modify the `main_method()` function to:
  - Adjust the number of layers (`no_of_layers`) and neurons per layer (`neurons_per_layer`).
  - Enable or disable preprocessing, t-SNE visualization, and verbose outputs.
  - Change learning rate and number of epochs.

## Example Workflow

1. Load the dataset and preprocess images using mean subtraction.
2. Define a network architecture with three layers and train on the dataset.
3. Evaluate the model on test data, visualize t-SNE representations, and plot the confusion matrix.
4. Save the trained model for later use.

## Visualization

During training, the script plots:
- Training loss and accuracy curves.
- t-SNE representations of the dataset and hidden layer activations.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any enhancements or bug fixes.

