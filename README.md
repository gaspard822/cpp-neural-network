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

Everything is implemented in an object-oriented manner. The architecture of the project is the following:

**NeuralNetwork** is the backbone of the project. It is assigned an optimizer and a loss function and stores the layers of the network. It contains functions like `forward()`, `backward()`, `infer()` and `train()`.

**FullyConnectedLayer** is a subclass of **Layer**, and as its name suggests, it implements a fully connected layer, with the functions `forward()`, `backward()` and `infer()`.

**ActivationFunction** is the supertype of **Relu**, **Sigmoid** and **Identity**. Its interface simply consists of the two functions `apply()` and `derivative()`.

**LossFunction** is the supertype of **MeanSquaredError** and **CrossEntropy**. Its interface consists of the two functions `compute()` and `derivative()`.

**Optimizer** is the supertype of **AdamOptimizer** and **VanillaSGDOptimizer**. The main function they implement is `update_gradients()`.

More details can be found in the source code comments.

#### Forwarding
Let $a^{(L)} \in \R^n$ be the input of layer $L$. It first gets normalized as follows:

$$\hat{a}^{(L)} = \frac{a^{(L)}-\mu^{(L)}}{\sqrt{{\sigma^{(L)}}^2}+\varepsilon}$$

where $\mu^{(L)},{\sigma^{(L)}}^2 \in \R^n$ are the mean resp. the variance of all inputs that this layer receives, and $\varepsilon \in \R^n$ is a vector of small values ensuring numerical stability (we use an element-wise division operator here).

We then scale and shift this normalized vector by the parameters $\gamma^{(L)} \in \R^n$ and $\beta^{(L)} \in \R^n$ of layer $L$:

$$\bar{a}^{(L)} = \hat{a}^{(L)} \odot \gamma^{(L)} + \beta^{(L)}$$

where $\odot$ denotes the element-wise multiplication.

Next, we multiply $\bar{a}^{(L)}$ by the weights and add the bias:

$$z^{(L)} = W^{(L)}\bar{a}^{(L)} + b^{(L)}$$

where $W^{(L)} \in \R^{m \times n}$ and $b^{(L)} \in \R^m$ are the weights respectively the bias of layer $L$.

Finally, we apply the activation function $\sigma:\R^m \to \R^m$:

$$a^{(L)} = \sigma(z^{(L)})$$

Note: Each layer keeps a "running mean" and a "running variance", that get updated as follows every time data is forwarded and that are used for normalizing new data when doing inference.
$$
\begin{align*}
    \mu_\text{running}^{(L)} &= \text{momentum} \cdot \mu_\text{running}^{(L)} + (1-\text{momentum})\mu^{(L)} \\
    \sigma_\text{running}^{(L)} &= \text{momentum} \cdot \sigma_\text{running}^{(L)} + (1-\text{momentum})\sigma^{(L)}
\end{align*}
$$
where $\text{momentum} \in (0,1)$.

#### Backpropagation
Let $\mathcal{L}$ be the loss function. We use the notation $dy$ instead of $\frac{\partial \mathcal{L}}{\partial y}$. Provided we receive the gradient from the next layer, $da^{(L+1)}$, we propagate the gradients through layer $L$ as follows:

$$
\begin{align*}
    dz^{(L)} &= da^{(L+1)} \odot \sigma'(z^{(L)}) \\
    dW^{(L)} &= dz^{(L)} {\bar{a}^{(L)}}^T \\
    db^{(L)} &= dz^{(L)} \\
    d\bar{a}^{(L)} &= {dW^{(L)}}^T dz^{(L)} \\
    d\gamma^{(L)} &= d\bar{a}^{(L)} \odot \hat{a}^{(L)} \\
    d\beta^{(L)} &= d\bar{a}^{(L)}\\
    d\hat{a}^{(L)} &= d\bar{a}^{(L)} \odot \gamma^{(L)} \\
    da^{(L)} &= d\hat{a}^{(L)} \odot \frac{1}{\sqrt{{\sigma^{(L)}}^2 + \varepsilon}}
\end{align*}
$$

#### Derivative of the MSE loss
The derivative of the MSE loss is straightforward. The mean squared error (MSE) loss for a batch of $m$ samples is defined as:

$$
\text{MSE}(\mathbf{Y}_{\text{true}}, \mathbf{Y}_{\text{pred}}) = \frac{1}{m} \sum_{i=1}^{m} \| y_{\text{true}}^{(i)} - y_{\text{pred}}^{(i)} \|^2
$$

where $y_{\text{true}}^{(i)}, y_{\text{pred}}^{(i)} \in \mathbb{R}^n$ are the true and predicted vectors for the $i$-th sample in the batch.

Taking the derivative with respect to one sample gives:

$$
\frac{\partial \text{MSE}}{\partial y_{\text{pred}}^{(i)}} = \frac{2}{m} \left( y_{\text{pred}}^{(i)} - y_{\text{true}}^{(i)} \right)
$$

So the gradient for the full batch is the matrix:

$$
\frac{\partial \text{MSE}}{\partial \mathbf{Y}_{\text{pred}}} = \frac{2}{m} \left( \mathbf{Y}_{\text{pred}} - \mathbf{Y}_{\text{true}} \right)
$$

#### Derivative of the cross-entropy loss
The cross-entropy loss for a batch of $m$ samples is defined as:

$$
\text{CELoss}(\mathbf{Z}, \mathbf{Y})
= -\frac{1}{m}\sum_{i=1}^{m} \sum_{j=1}^{n} y^{(i)}_j \log \left( \frac{e^{z^{(i)}_j}}{\sum_{k=1}^{n}e^{z^{(i)}_k}} \right)
= \frac{1}{m}\sum_{i=1}^{m} \sum_{j=1}^{n} y^{(i)}_j \left( -z^{(i)}_j + \log \left( \sum_{k=1}^{n}e^{z^{(i)}_k} \right) \right)
$$

where $z^{(i)},y^{(i)} \in \R^n$ are the logits and one-hot encoded labels for the $i$-th sample in the batch. The first derivation is here for numerical stability, since it avoids dividing by very small numbers or taking the logarithm of values close to 0. To further enhance the computational stability, we leverage the fact that:

$$
\log \left( {\sum_{k=1}^{n}e^{z^{(i)}_k}} \right)
= \log \left( e^{z^{(i)}_\text{max}}\sum_{k=1}^{n}e^{z^{(i)}_k - z^{(i)}_\text{max}} \right)
= z^{(i)}_\text{max} + \log \left( \sum_{k=1}^{n}e^{z^{(i)}_k - z^{(i)}_\text{max}} \right)
$$

where $z^{(i)}_\text{max}$ is the largest logit value for sample $i$, because now, we reduce the risk of computing the exponential of a large number. We now have:

$$
\text{CELoss}(\mathbf{Z}, \mathbf{Y})
= \frac{1}{m}\sum_{i=1}^{m} \sum_{j=1}^{n} y^{(i)}_j \left( -z^{(i)}_j + z^{(i)}_\text{max} + \log \left( \sum_{k=1}^{n}e^{z^{(i)}_k - z^{(i)}_\text{max}} \right) \right)
$$

This formula is the one that we use in the code when computing the cross-entropy loss of our network on some dataset.

For the $i$-th sample, when taking the derivative with respect to the $j$-th logit, $z^{(i)}_j$, we have the two following cases.

Case $y_j^{(i)} = 1$:
$$
\begin{align*}
& \frac{\partial}{\partial z^{(i)}_j}\sum_{j'=1}^{n} y^{(i)}_{j'} \left( -z^{(i)}_{j'} + \log \left( \sum_{k=1}^{n}e^{z^{(i)}_k} \right) \right) \\
&= \frac{\partial}{\partial z^{(i)}_j} \left( -z^{(i)}_j + \log \left( \sum_{k=1}^{n}e^{z^{(i)}_k} \right) \right) \\
&= -1 + \frac{e^{z^{(i)}_j}}{\sum_{k=1}^{n}e^{z^{(i)}_k}}
\end{align*}
$$

Case $y_l^{(i)} = 1, l \neq j$:
$$
\begin{align*}
& \frac{\partial}{\partial z^{(i)}_j}\sum_{j'=1}^{n} y^{(i)}_{j'} \left( -z^{(i)}_{j'} + \log \left( \sum_{k=1}^{n}e^{z^{(i)}_k} \right) \right) \\
&= \frac{\partial}{\partial z^{(i)}_j} \left( -z^{(i)}_l + \log \left( \sum_{k=1}^{n}e^{z^{(i)}_k} \right) \right) \\
&= \frac{e^{z^{(i)}_j}}{\sum_{k=1}^{n}e^{z^{(i)}_k}}
\end{align*}
$$

Therefore, for the final derivative of the cross-entropy loss function of the $i$-th sample with respect to $z^{(i)}_j$, we get:

$$
\frac{\partial \text{CELoss}(\mathbf{Z}, \mathbf{Y})}{\partial z^{(i)}_j}
= \frac{1}{m} \left( \frac{e^{z^{(i)}_j}}{\sum_{k=1}^{n}e^{z^{(i)}_k}} - y^{(i)}_j \right)
$$

For better numerical stability, we use the same trick as before once again and we get the following formula, that is actually implemented in the code.

$$
\frac{\partial \text{CELoss}(\mathbf{Z}, \mathbf{Y})}{\partial z^{(i)}_j}
= \frac{1}{m} \left( \frac{e^{z^{(i)}_j - z^{(i)}_\text{max}}}{\sum_{k=1}^{n}e^{z^{(i)}_k - z^{(i)}_\text{max}}} - y^{(i)}_j \right)
$$

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