# MLP from scratch in C++

This is a project that implements a modular neural network framework in C++ from scratch, using Eigen. The goal is certainly not to build a competitive machine learning framework, but rather to deepen my understanding by implementing things myself.

### Features

- Multilayer perceptron (structured in a way that allows for extension to more general architectures)
- Training with backpropagation
- Reduces unnecessary data copying for better performance
- Supports multiple optimizers (Adam, SGD)
- Includes loss functions such as MSE and Cross-Entropy
- Supports activation functions like ReLU and sigmoid
- Very modular - adding new layer types, optimizers, loss functions, or activation functions is straightforward
- Batch normalization
- Early stopping

### Installation

#### Requirements

- C++11 or later (project currently uses the C++17 standard)
- CMake ≥ 3.10
- [Eigen](https://eigen.tuxfamily.org/) (header-only; v3.4+ recommended)

#### Build Instructions

```bash
git clone https://github.com/gaspard822/cpp-neural-network.git
cd cpp-neural-network
mkdir build && cd build
cmake ..
make
```

### Usage

The following block of code displays a template of how to create an MLP that:
* uses the mean squared error as a loss function
* uses Adam as an optimizer
* has two hidden fully connected layers: the first one has 64 units and uses the ReLU activation function, while the second one has 16 units and uses the sigmoid activation function
* gets trained with the training set `(X_train, Y_val)` and the validation set `(X_val, Y_val)`
* gets trained with a batch size of 128
* has a number of epochs bounded to 300 (can stop before due to early stopping)

```C++
NeuralNetwork nn("MeanSquaredError", "Adam");
FullyConnectedLayer* layer_1 = new FullyConnectedLayer(new Relu(), 128, 64);
nn.add_layer(layer_1);
FullyConnectedLayer* layer_2 = new FullyConnectedLayer(new Sigmoid(), 64, 16);
nn.add_layer(layer_2);
FullyConnectedLayer* layer_3 = new FullyConnectedLayer(new Identity(), 16, 1);
nn.add_layer(layer_3);
nn.train(X_train, Y_train, 300, 128, X_val, Y_val);
```

To use cross-entropy loss instead of mean squared error, replace `"MeanSquaredError"` with `"CrossEntropy"`.
To use vanilla SGD instead of Adam, replace `"Adam"` with `"VanillaSGD"`.

### Example

The network was tested on the MNIST dataset provided at  
https://www.kaggle.com/competitions/digit-recognizer/data.

The specification and training code can be found in [`src/mnist.cpp`](src/mnist.cpp). The model uses the cross-entropy loss function and the Adam optimizer.

It consists of three hidden fully connected layers with ReLU activation with the following sizes: 512, 256, and 128 units. This results in a total of 570'794 trainable parameters.

The labeled dataset contains 42'000 samples, which were split as follows:
- 34'000 for training
- 4'000 for validation
- 4'000 for testing

The best performance was obtained by training the network for 300 epochs with a batch size of 1'024, and without early stopping.

The final model achieved 96% accuracy on the unlabeled Kaggle test set.  
Training took approximately 95 seconds on a standard CPU.

The model is available in the repository at models/final_model.txt.

### Technical details

For detailed explanations, equations, and implementation notes, please refer to [technical_details.pdf](technical_details.pdf).

### Future work
Potential future improvements include:
- Dropout regularization
- L1/L2 regularization
- Parallelization
- GPU acceleration (possibly via a graphics API)
- Support for additional architectures

### License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.