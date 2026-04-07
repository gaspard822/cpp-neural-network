#include <iostream>
#include <fstream>
#include <chrono>
#include <random>
#include <mlx/mlx.h>
#include "mlp/mnist.cpp"
#include "transformer/translation.cpp"

using namespace std;
namespace mx = mlx::core;

int main() {

    // train_test_mnist();
    // infer_mnist();

    TrainingConfig cfg;
    init_transformer_model(cfg);
    // train_transformer_model(cfg, true, false);

    return 0;
}
