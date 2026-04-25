# Neural Networks from Scratch in C++

A from-scratch implementation of neural networks in C++, with no reliance on machine learning frameworks. All forward passes, backward passes, gradient computations and parameter updates are implemented manually.

The project was built in two stages:

1. **Multilayer Perceptron:** A simple modular MLP framework with backpropagation, batch normalization, early stopping, and support for multiple optimizers (Adam, SGD) and loss functions (cross-entropy, MSE). Tested on MNIST digit classification (96% accuracy).

2. **Transformer:** A full encoder-decoder transformer with multi-head attention, BPE tokenization, and Pre-LN residual connections. Trained on English-to-French translation using ~370k sentence pairs from the OPUS corpus. Produces correct translations on most simple sentences. See the [technical details](docs/transformer.pdf).

Both stages use Apple's [MLX](https://github.com/ml-explore/mlx) framework for GPU-accelerated matrix operations on Apple Silicon.

## Build

```bash
git clone https://github.com/gaspard822/cpp-neural-network.git
cd cpp-neural-network
mkdir build && cd build
cmake ..
make
```

Requirements: C++17, CMake >= 3.10, MLX.

## Technical Details

For a detailed write-up covering the transformer's architecture, mathematical derivations (forward and backward passes for every layer), training procedure and results, see:

**[Transformer - Technical Details (PDF)](docs/transformer.pdf)**

A shorter document covering the MLP is also available: [MLP - Technical Details (PDF)](docs/mlp.pdf)

## License

MIT - see [LICENSE](LICENSE).